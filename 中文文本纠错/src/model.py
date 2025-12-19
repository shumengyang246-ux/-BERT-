# 纠错和检测模型搭建
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import Optional, Dict, Any, Tuple
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

class SoftMaskedOutput:
    """
    Standardized outputs for training/inference.
    """
    def __init__(self,detect_logits, detect_prob, corr_logits, input_embed, soft_word_emb, hidden):
        self.detect_logits=detect_logits  # [B, L]
        self.detect_prob=detect_prob        # [B, L]
        self.corr_logits=corr_logits        # [B, L, V]
    # 有用的中间张量（可选择在外部使用）
        self.input_embed=input_embed        # [B, L, H] BERT 嵌入(input_ids, token_type_ids)
        self.soft_word_emb=soft_word_emb      # [B, L, H] soft-masked 词嵌入
        self.hidden=hidden                  # [B, L, H] 纠正网络隐藏层

# 搭建错别字检测网络
class DetectionNetwork(nn.Module):
    """
    检测网络: BiGRU -> linear -> per-position logits.
    输入是 BERT 的 “输入嵌入”（词 + 位置 + 片段，然后是层归一化 /dropout），
    [B, L, H]
    输出: logits [B, L], prob [B, L]
    """
    def __init__(self, hidden_size: int, gru_hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * gru_hidden, 1)  # -> logits per position

    def forward(
        self,
        input_embed: torch.Tensor,           # [B, L, H]
        attention_mask: Optional[torch.Tensor] = None,  # [B, L], optional
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          detect_logits: [B, L]
          detect_prob:   [B, L]
        """
        #  GRU 前向传播。稍后使用 attention_mask 进行损失掩码操作。
        if attention_mask is None:     # 退化到原实现
            out, _ = self.gru(input_embed)
        else:
            lengths = attention_mask.sum(dim=1).to(torch.long).cpu()  # [B]
            packed = pack_padded_sequence(input_embed, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_embed.size(1)) # [B, L, 2*gru_hidden]
        out = self.dropout(out)
        logits = self.fc(out).squeeze(-1)     # [B, L]
        prob = torch.sigmoid(logits)          # [B, L]
        return logits, prob

# 软掩码加入
class SoftMask(nn.Module):
    """
    在词嵌入（不包括位置 / 段落嵌入）上进行混合，避免在将 inputs_embeds 传入 BertModel 时重复添加位置 / 段落信息。
    """
    def __init__(self, word_embeddings: nn.Embedding, mask_token_id: int = 103):
        super().__init__()
        self.word_embeddings = word_embeddings
        self.mask_token_id = int(mask_token_id)

    def forward(
        self,
        input_ids: torch.Tensor,     # [B, L]
        detect_prob: torch.Tensor,   # [B, L], p_i in [0, 1]
    ) -> torch.Tensor:
        """
        返回:
          soft_word_emb: [B, L, H]
        """
        word_emb = self.word_embeddings(input_ids)  # [B, L, H]
        # [MASK] 嵌入向量来自同一个嵌入表（作为 BERT 的一部分可训练）
        mask_vec = self.word_embeddings.weight[self.mask_token_id]  # [H]
        mask_emb = mask_vec.view(1, 1, -1).expand_as(word_emb)      # [B, L, H]

        p = detect_prob.unsqueeze(-1)  # [B, L, 1]
        soft_word_emb = p * mask_emb + (1.0 - p) * word_emb
        return soft_word_emb


class CorrectionNetwork(nn.Module):
    """
    修正网络：
       在软掩码嵌入（inputs_embeds）上运行 BERT
       与原始输入嵌入的残差连接
       词汇表投影，用于预测每个位置的修正 token
    """
    def __init__(self, bert: BertModel, dropout: float = 0.1):
        super().__init__()
        self.bert = bert
        hidden = bert.config.hidden_size
        vocab = bert.config.vocab_size

        self.dropout = nn.Dropout(dropout)
        self.mlm_head = BertOnlyMLMHead(bert.config)

    def forward(
        self,
        soft_word_emb: torch.Tensor,          # [B, L, H] (word-only soft embeddings)
        attention_mask: torch.Tensor,         # [B, L]
        token_type_ids: Optional[torch.Tensor],
        residual_input_embed: torch.Tensor,   # [B, L, H] full embedding from bert.embeddings(...)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回:
          corr_logits: [B, L, V]
          hidden:      [B, L, H] (after residual)
        """
        out = self.bert(
            inputs_embeds=soft_word_emb,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        h_c = out.last_hidden_state             # [B, L, H]
        h0 = h_c + residual_input_embed         # residual
        h0 = self.dropout(h0)
        corr_logits = self.mlm_head(h0)        # [B, L, V]
        return corr_logits, h0

class SoftMaskedBertForCSC(nn.Module):
    """
    完整结构:
      input_ids -> bert.embeddings -> DetectionNetwork -> p
      input_ids + p -> SoftMask -> soft_word_emb
      soft_word_emb -> BERT -> residual -> vocab logits
    """
    def __init__(
        self,
        bert_name_or_path: str = "bert-base-chinese",
        gru_hidden: int = 256,
        mask_token_id: int = 103,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name_or_path)

        hidden = self.bert.config.hidden_size
        self.detector = DetectionNetwork(hidden_size=hidden, gru_hidden=gru_hidden, dropout=dropout)

        # 使用 BERT 自身的嵌入表，以便梯度能够连贯地流动
        self.softmask = SoftMask(self.bert.embeddings.word_embeddings, mask_token_id=mask_token_id)

        self.corrector = CorrectionNetwork(self.bert, dropout=dropout)

    def forward(
        self,
        input_ids: torch.Tensor,                 # [B, L]
        attention_mask: torch.Tensor,            # [B, L]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, L] or None
    ) -> SoftMaskedOutput:
        """
        总体网络前向传播
        """
        # 1) 完整的输入嵌入（词 + 词性 + 分词→层归一化 /dropout），用于检测和残差。
        input_embed = self.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )  # [B, L, H]

        # 2) 检测网络：每个位置的错误概率 p_i
        detect_logits, detect_prob = self.detector(input_embed, attention_mask=attention_mask)

        # 3) Soft-masking: 在词嵌入上进行混合，以避免重复添加位置 / 段落信息。
        soft_word_emb = self.softmask(input_ids=input_ids, detect_prob=detect_prob)  # [B, L, H]

        # 4) 校正网络（BERT）+ 残差 + 词汇投影
        corr_logits, hidden = self.corrector(
            soft_word_emb=soft_word_emb,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            residual_input_embed=input_embed,
        )

        return SoftMaskedOutput(
            detect_logits=detect_logits,
            detect_prob=detect_prob,
            corr_logits=corr_logits,
            input_embed=input_embed,
            soft_word_emb=soft_word_emb,
            hidden=hidden,
        )


