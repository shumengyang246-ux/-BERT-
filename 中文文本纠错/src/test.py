# 交互式测试模型能力
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast
from model import SoftMaskedBertForCSC
from config import Config

CKPT_PATH = r"C:\Users\dell\Desktop\项目实战\Soft-Mask-Bert\final_model\ckpt_last.pt"

def is_cjk_char(ch: str) -> bool:
    """粗略判断是否为中日韩统一表意文字（单字符）"""
    if not isinstance(ch, str) or len(ch) != 1:
        return False
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
    )


class CSCInference:
    def __init__(
        self,
        ckpt_path: str,
        device: str | None = None,
        tau: float = 0.32,                 # 1) 新增：detect阈值
        top_k_edits: int = 3,             # 4) 新增：一句最多改Top-K处
        cand_topk: int = 8,               # 3) 候选约束：只看纠错头Top-N候选
        min_logit_margin: float = 1.0,    # 3) 候选约束：候选logit需明显高于原字
        restrict_cjk: bool = True,        # 3) 候选约束：中文仅改成中文
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = float(tau)
        self.top_k_edits = int(top_k_edits)
        self.cand_topk = int(cand_topk)
        self.min_logit_margin = float(min_logit_margin)
        self.restrict_cjk = bool(restrict_cjk)

        print(f"[Info] Loading model from: {ckpt_path}")
        print(f"[Info] Device: {self.device}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # 2) 恢复配置：不再硬编码 BERT_MODEL_NAME（你原脚本这里是硬编码的 :contentReference[oaicite:4]{index=4}）
        saved_config_dict = checkpoint.get("config", {})
        valid_keys = Config.__dataclass_fields__.keys()
        filtered_dict = {k: v for k, v in saved_config_dict.items() if k in valid_keys}
        self.cfg = Config(**filtered_dict)

        # tokenizer/model 名称优先使用 checkpoint/config
        tokenizer_name = getattr(self.cfg, "tokenizer_name", None) or getattr(self.cfg, "bert_name", None) or "bert-base-chinese"
        bert_name = getattr(self.cfg, "bert_name", None) or tokenizer_name

        print(f"[Info] tokenizer_name = {tokenizer_name}")
        print(f"[Info] bert_name      = {bert_name}")

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
        print("[Info] Model loaded successfully!\n")

    def _encode_char_level(self, text: str):
        """字符级编码：与训练集对齐（dataset.py同样是list(src)逐字转id再加CLS/SEP :contentReference[oaicite:5]{index=5}）"""
        text = text.strip()
        if not text:
            return None

        max_body = int(self.cfg.max_length) - 2
        chars = list(text)[:max_body]

        # convert_tokens_to_ids 对未知字符会给unk_token_id
        ids = self.tokenizer.convert_tokens_to_ids(chars)
        ids = [x if x is not None else self.tokenizer.unk_token_id for x in ids]

        input_ids = [self.tokenizer.cls_token_id] + ids + [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        return (
            torch.tensor([input_ids], dtype=torch.long, device=self.device),
            torch.tensor([attention_mask], dtype=torch.long, device=self.device),
            torch.tensor([token_type_ids], dtype=torch.long, device=self.device),
            chars,  # 原始字符（不含CLS/SEP）
        )

    @torch.no_grad()
    def predict(self, text: str):
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

        # [L]
        detect_probs = out.detect_prob[0].detach().cpu()     # torch.Tensor
        corr_logits = out.corr_logits[0].detach().cpu()      # [L, V]
        pred_ids = corr_logits.argmax(dim=-1)                # [L]

        # 将输入ids也取出来（用于取“原字符对应的token id”）
        input_ids_cpu = input_ids[0].detach().cpu()          # [L]

        # 生成“候选改动池”（先不应用Top-K预算）
        proposals = []
        seq_len = input_ids_cpu.size(0)

        # 有效范围：1..L-2 对应 src_chars 的 0..len-1
        for pos in range(1, seq_len - 1):
            p_err = float(detect_probs[pos].item())

            # 1) tau 门控：检测概率低于阈值，不考虑修改
            if p_err <= self.tau:
                continue

            src_id = int(input_ids_cpu[pos].item())
            src_char = src_chars[pos - 1]

            # top-N 候选（含原字/预测字）
            topv = min(self.cand_topk, corr_logits.size(-1))
            top_scores, top_ids = torch.topk(corr_logits[pos], k=topv, dim=-1)

            # 原字符 logit（用于 margin 约束）
            src_logit = float(corr_logits[pos, src_id].item())

            best = None
            for cand_id, cand_logit in zip(top_ids.tolist(), top_scores.tolist()):
                cand_id = int(cand_id)
                cand_logit = float(cand_logit)
                if cand_id == src_id:
                    continue



                cand_tok = self.tokenizer.convert_ids_to_tokens([cand_id])[0]

                # 过滤 special / unk / pad 等
                if cand_tok in ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]:
                    continue
                # WordPiece残留
                cand_tok = cand_tok.replace("##", "")

                # 只允许单字符替换（更贴合你的训练标注）
                if len(cand_tok) != 1:
                    continue

                # 3) 候选约束：中文仅改成中文（可开关）
                if self.restrict_cjk and is_cjk_char(src_char):
                    if not is_cjk_char(cand_tok):
                        continue

                # 3) 候选约束：logit margin（候选必须明显优于原字）
                if (cand_logit - src_logit) < self.min_logit_margin:
                    continue

                best = (cand_tok, cand_id, cand_logit)
                break

            if best is None:
                continue

            pred_char, pred_id, pred_logit = best
            if pred_char == src_char:
                continue

            # proposal 评分：优先考虑高检测概率，其次考虑替换“收益”(margin)
            score = p_err * (pred_logit - src_logit)

            proposals.append({
                "pos": pos,                      # token位置（含CLS偏移）
                "char_index": pos - 1,           # 字符位置
                "src_char": src_char,
                "pred_char": pred_char,
                "p_err": p_err,
                "margin": float(pred_logit - src_logit),
                "score": float(score),
            })

        # 4) 改动预算 Top-K：只应用最可信的K个
        proposals.sort(key=lambda x: x["score"], reverse=True)
        chosen = proposals[: max(0, self.top_k_edits)]

        # 应用改动
        corrected = src_chars[:]  # list[str]
        details = []
        for item in chosen:
            idx = item["char_index"]
            corrected[idx] = item["pred_char"]
            details.append(
                f"位置 {idx}: '{item['src_char']}' -> '{item['pred_char']}' "
                f"(错误概率: {item['p_err']:.4f}, margin: {item['margin']:.2f})"
            )

        final_text = "".join(corrected)
        return final_text, details


def main():
    try:
        # 你也可以把 tau/top_k_edits 等改成命令行参数；这里先给合理默认
        engine = CSCInference(
            CKPT_PATH,
            tau=0.32,
            top_k_edits=3,
            cand_topk=8,
            min_logit_margin=1.0,
            restrict_cjk=True,
        )
    except Exception as e:
        print(f"[Error] {e}")
        return

    print("=" * 60)
    print("Soft-Masked BERT 纠错演示（改进解码：tau/候选约束/Top-K）")
    print("提示：输入 'q' 或 'exit' 退出程序")
    print("=" * 60)

    while True:
        try:
            raw_text = input("\n请输入句子: ").strip()
            if raw_text.lower() in ["q", "exit"]:
                print("Bye!")
                break
            if not raw_text:
                continue

            corrected_text, logs = engine.predict(raw_text)

            print("-" * 30)
            print(f"原句: {raw_text}")
            print(f"纠错: {corrected_text}")

            if logs:
                print("修改详情:")
                for log in logs:
                    print(f"  * {log}")
            else:
                print("  (无修改)")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"[Error during prediction] {e}")


if __name__ == "__main__":
    main()
