# 使用Soft-Mask-Bert模型进行中文拼写纠错的推理模块
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import BertTokenizerFast

from config import Config
from model import SoftMaskedBertForCSC

def is_cjk_char(ch: str) -> bool:
    if not isinstance(ch, str) or len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )

class CSCInference:
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        tau: float = 0.32, # 检测错别字概率阈值
        top_k_edits: int = 3, # 一句话中最多修改的错误数
        cand_topk: int = 8, # 候选约束：只看纠错头Top-N候选
        min_logit_margin: float = 1.0, # 候选约束：候选logit需明显高于原字
        restrict_cjk: bool = True,
    ) -> None:
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = float(tau)
        self.top_k_edits = int(top_k_edits)
        self.cand_topk = int(cand_topk)
        self.min_logit_margin = float(min_logit_margin)
        self.restrict_cjk = bool(restrict_cjk)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        saved_config_dict = checkpoint.get("config", {})
        valid_keys = Config.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in saved_config_dict.items() if k in valid_keys}
        self.cfg = Config(**filtered_dict)

        tokenizer_name = getattr(self.cfg, "tokenizer_name", None) or getattr(self.cfg, "bert_name", None) or "bert-base-chinese"
        bert_name = getattr(self.cfg, "bert_name", None) or tokenizer_name

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.model = SoftMaskedBertForCSC(
            bert_name_or_path=bert_name,
            gru_hidden=self.cfg.gru_hidden,
            mask_token_id=self.tokenizer.mask_token_id,
            dropout=0.0,
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def _encode_char_level(
        self, text: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]]:
        text = text.strip()
        if not text:
            return None

        max_body = int(self.cfg.max_length) - 2
        chars = list(text)[:max_body]

        ids = self.tokenizer.convert_tokens_to_ids(chars)
        ids = [x if x is not None else self.tokenizer.unk_token_id for x in ids]

        input_ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return (
            torch.tensor([input_ids], dtype=torch.long, device=self.device),
            torch.tensor([attention_mask], dtype=torch.long, device=self.device),
            torch.tensor([token_type_ids], dtype=torch.long, device=self.device),
            chars,
        )

    @torch.no_grad()
    def predict(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        if not text.strip():
            return text, []

        encoded = self._encode_char_level(text)
        if encoded is None:
            return text, []

        input_ids, attention_mask, token_type_ids, src_chars = encoded

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        detect_probs = out.detect_prob[0].detach().cpu()
        corr_logits = out.corr_logits[0].detach().cpu()
        input_ids_cpu = input_ids[0].detach().cpu()

        proposals: List[Dict[str, Any]] = []
        seq_len = input_ids_cpu.size(0)

        for pos in range(1, seq_len - 1):
            p_err = float(detect_probs[pos].item())
            if p_err <= self.tau:
                continue

            src_id = int(input_ids_cpu[pos].item())
            src_char = src_chars[pos - 1]

            topv = min(self.cand_topk, corr_logits.size(-1))
            top_scores, top_ids = torch.topk(corr_logits[pos], k=topv, dim=-1)
            src_logit = float(corr_logits[pos, src_id].item())

            best = None
            for cand_id, cand_logit in zip(top_ids.tolist(), top_scores.tolist()):
                cand_id = int(cand_id)
                cand_logit = float(cand_logit)
                if cand_id == src_id:
                    continue

                cand_tok = self.tokenizer.convert_ids_to_tokens([cand_id])[0]
                if cand_tok in ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
                    continue
                cand_tok = cand_tok.replace("##", "")
                if len(cand_tok) != 1:
                    continue
                if self.restrict_cjk and is_cjk_char(src_char) and not is_cjk_char(cand_tok):
                    continue
                if (cand_logit - src_logit) < self.min_logit_margin:
                    continue

                best = (cand_tok, cand_id, cand_logit)
                break

            if best is None:
                continue

            pred_char, pred_id, pred_logit = best
            if pred_char == src_char:
                continue

            score = p_err * (pred_logit - src_logit)
            proposals.append(
                {
                    "pos": pos,
                    "index": pos - 1,
                    "src_char": src_char,
                    "pred_char": pred_char,
                    "p_err": p_err,
                    "margin": float(pred_logit - src_logit),
                    "score": float(score),
                }
            )

        proposals.sort(key=lambda x: x["score"], reverse=True)
        chosen = proposals[: max(0, self.top_k_edits)]

        corrected = src_chars[:]
        edits: List[Dict[str, Any]] = []
        for item in chosen:
            idx = item["index"]
            corrected[idx] = item["pred_char"]
            edits.append(
                {
                    "type": "replace",
                    "index": idx,
                    "src_char": item["src_char"],
                    "pred_char": item["pred_char"],
                    "p_err": item["p_err"],
                    "margin": item["margin"],
                    "score": item["score"],
                }
            )

        final_text = "".join(corrected)
        return final_text, edits
