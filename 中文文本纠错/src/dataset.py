# 数据处理模块
"""
主要对齐model.py中定义的输入格式
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

IGNORE_INDEX = -100

def _safe_convert_tokens_to_ids(tokenizer: BertTokenizerFast, tokens: List[str]) -> List[int]:
    """
      将单字符标记转换为标识符。
      未知字符将映射到 [UNK]。
    """
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # HF可能返回 list[int]；通常未知字符会直接给 unk_token_id
    # 这里保证都是 int
    unk = tokenizer.unk_token_id
    fixed = []
    for x in ids:
        if x is None:
            fixed.append(unk)
        else:
            fixed.append(int(x))
    return fixed


class SoftMaskedJsonlDataset(Dataset):
    """
    读取 JSONL（每行一个 JSON），做“字符级”tokenization，并构造两类监督：
      - labels_correct: 纠正标签（多类分类），按位置给出 tgt token id
      - labels_detect : 检测标签（二分类），来自 detection_labels (g_i)

    输出字段（与训练脚本/模型对接常用）：
      input_ids        [L]
      attention_mask   [L]
      token_type_ids   [L]
      labels_correct   [L] (IGNORE_INDEX for CLS/SEP/PAD)
      labels_detect    [L] (IGNORE_INDEX for CLS/SEP/PAD)
      src_text/tgt_text 便于debug
    """
    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer_name: str = "bert-base-chinese",
        max_length: int = 128,
        drop_length_mismatch: bool = False, # 是否丢弃长度不匹配的样本
        return_text: bool = False,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
        self.max_length = int(max_length)
        self.drop_length_mismatch = bool(drop_length_mismatch)
        self.return_text = bool(return_text)

        # 读入并保存样本，这里按行处理
        self.samples: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                src = obj.get("src")
                tgt = obj.get("tgt")
                det = obj.get("detection_labels")

                if not isinstance(src, str) or not isinstance(tgt, str) or not isinstance(det, list):
                    continue

                if len(src) != len(tgt) or len(det) != len(src):
                    if self.drop_length_mismatch:
                        continue # 未对齐的样本进行丢弃
                    else:
                        # 保留全部样本
                        pass                     
                self.samples.append(obj)

        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {self.path}")
        print(f"Loaded {len(self.samples)} samples from {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_one(self, src: str, tgt: str, det_labels: List[int]) -> Dict[str, torch.Tensor]:
        """
        核心编码逻辑：
        1) 按字符切分 src/tgt（保证与 detection_labels 对齐）
        2) 转为 token id
        3) 拼接 [CLS] ... [SEP]
        4) 构造 labels，并在 CLS/SEP 处置为 IGNORE_INDEX
        5) 截断到 max_length
        """
        # 预留 [CLS] 和 [SEP]
        max_body = self.max_length - 2
        if len(src) > max_body:
            src = src[:max_body]
            tgt = tgt[:max_body]
            det_labels = det_labels[:max_body]

        src_tokens = list(src)
        tgt_tokens = list(tgt)

        src_ids = _safe_convert_tokens_to_ids(self.tokenizer, src_tokens)
        tgt_ids = _safe_convert_tokens_to_ids(self.tokenizer, tgt_tokens)

        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)

        input_ids = [cls_id] + src_ids + [sep_id]
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        # labels：在 CLS/SEP 上置 IGNORE_INDEX
        labels_correct = [IGNORE_INDEX] + tgt_ids + [IGNORE_INDEX]
        labels_detect = [IGNORE_INDEX] + [int(x) for x in det_labels] + [IGNORE_INDEX]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels_correct": torch.tensor(labels_correct, dtype=torch.long),
            "labels_detect": torch.tensor(labels_detect, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.samples[idx]
        src: str = obj["src"]
        tgt: str = obj["tgt"]
        det: List[int] = obj["detection_labels"]

        encoded = self._encode_one(src, tgt, det)

        if self.return_text:
            encoded["src_text"] = src
            encoded["tgt_text"] = tgt

        return encoded


def collate_softmasked_batch(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, Any]:
    """
    DataLoader 的 collate_fn：对一个 batch 做动态 padding（pad 到 batch 内最长 L）。
    padding 规则：
      - input_ids: pad_token_id
      - attention_mask: 0
      - token_type_ids: 0
      - labels_correct / labels_detect: IGNORE_INDEX
    """
    max_len = max(x["input_ids"].size(0) for x in batch)

    def pad_1d(x: torch.Tensor, pad_value: int) -> torch.Tensor:
        if x.size(0) == max_len:
            return x
        pad_size = (0, max_len - x.size(0))
        return torch.nn.functional.pad(x, pad_size, value=pad_value)

    out: Dict[str, Any] = {}
    keys = ["input_ids", "attention_mask", "token_type_ids", "labels_correct", "labels_detect"]

    for k in keys:
        if k not in batch[0]:
            continue
        pad_value = {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "token_type_ids": 0,
            "labels_correct": IGNORE_INDEX,
            "labels_detect": IGNORE_INDEX,
        }[k]
        out[k] = torch.stack([pad_1d(x[k], pad_value) for x in batch], dim=0)

    # 把文本字段也拼回去（debug用）
    if "src_text" in batch[0]:
        out["src_text"] = [x["src_text"] for x in batch]
        out["tgt_text"] = [x["tgt_text"] for x in batch]

    return out
