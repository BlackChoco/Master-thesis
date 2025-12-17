import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    对比损失函数，支持三种InfoNCE变体和样本权重
    """
    def __init__(self, temperature: float = 0.07, loss_type: str = 'infonce',
                 infonce_mode: str = 'unidirectional', bidirectional_loss: bool = True):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.infonce_mode = infonce_mode
        self.bidirectional_loss = bidirectional_loss  # 新增: 控制in-batch模式的方向性
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        print(f"  ContrastiveLoss配置: 类型={loss_type}, InfoNCE模式={infonce_mode}, 温度={temperature}, 双向={bidirectional_loss}")

    def forward(self, anchor, positive, negatives=None, sample_weights=None):
        """
        计算对比损失

        Args:
            anchor: 锚点嵌入 [batch_size, embedding_dim]
            positive: 正样本嵌入 [batch_size, embedding_dim]
            negatives: 负样本嵌入 [batch_size, num_negatives, embedding_dim] (可选)
            sample_weights: 样本权重 [batch_size] (可选)

        Returns:
            加权对比损失
        """
        if anchor.nelement() == 0 or positive.nelement() == 0:
            return torch.tensor(0.0, device=anchor.device if anchor.nelement() > 0 else positive.device, requires_grad=True)

        if self.loss_type == 'infonce':
            if self.infonce_mode == 'unidirectional':
                return self._infonce_loss_unidirectional(anchor, positive, negatives, sample_weights)
            elif self.infonce_mode == 'bidirectional':
                return self._infonce_loss_bidirectional(anchor, positive, negatives, sample_weights)
            elif self.infonce_mode == 'in_batch':
                return self._infonce_loss_in_batch(anchor, positive, sample_weights)
            else:
                raise ValueError(f"不支持的 InfoNCE 模式: {self.infonce_mode}")
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
    
    def _infonce_loss_unidirectional(self, anchor, positive, negatives, sample_weights=None):
        """单向InfoNCE损失"""
        if negatives is None or negatives.nelement() == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        batch_size, num_negatives, _ = negatives.shape

        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        negatives_norm = F.normalize(negatives, dim=-1)

        pos_sim = self.cosine_sim(anchor_norm, positive_norm) / self.temperature

        anchor_expanded = anchor_norm.unsqueeze(1).expand(-1, num_negatives, -1)
        neg_sim = self.cosine_sim(anchor_expanded, negatives_norm) / self.temperature

        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

        # 计算损失
        if sample_weights is not None:
            # 使用样本权重的交叉熵损失
            loss = F.cross_entropy(all_sim, labels, reduction='none')
            weighted_loss = loss * sample_weights
            return weighted_loss.mean()
        else:
            return F.cross_entropy(all_sim, labels)
    
    def _infonce_loss_bidirectional(self, anchor, positive, negatives, sample_weights=None):
        """双向InfoNCE损失"""
        if negatives is None or negatives.nelement() == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        loss1 = self._compute_single_direction_loss(anchor, positive, negatives, sample_weights)
        loss2 = self._compute_single_direction_loss(positive, anchor, negatives, sample_weights)
        return (loss1 + loss2) / 2.0
    
    def _infonce_loss_in_batch(self, anchor, positive, sample_weights=None):
        """批内InfoNCE损失 - 支持双向/单向可配置"""
        batch_size = anchor.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)

        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)

        # 计算 anchor→positive 方向
        similarity_matrix_ap = torch.matmul(anchor_norm, positive_norm.t()) / self.temperature
        labels_ap = torch.arange(batch_size, device=anchor.device)
        loss_ap = F.cross_entropy(similarity_matrix_ap, labels_ap, reduction='none')

        if self.bidirectional_loss:
            # 双向: 同时计算 positive→anchor 方向
            similarity_matrix_pa = torch.matmul(positive_norm, anchor_norm.t()) / self.temperature
            labels_pa = torch.arange(batch_size, device=anchor.device)
            loss_pa = F.cross_entropy(similarity_matrix_pa, labels_pa, reduction='none')

            if sample_weights is not None:
                # 使用样本权重
                weighted_loss_ap = loss_ap * sample_weights
                weighted_loss_pa = loss_pa * sample_weights
                return (weighted_loss_ap.mean() + weighted_loss_pa.mean()) / 2.0
            else:
                return (loss_ap.mean() + loss_pa.mean()) / 2.0
        else:
            # 单向: 仅使用 anchor→positive 方向
            if sample_weights is not None:
                weighted_loss = loss_ap * sample_weights
                return weighted_loss.mean()
            else:
                return loss_ap.mean()

    def _compute_single_direction_loss(self, query, positive_key, negative_keys, sample_weights=None):
        """计算单向损失"""
        # query: [B, D], positive_key: [B, D], negative_keys: [B, N, D]
        batch_size, num_negatives, _ = negative_keys.shape

        query_norm = F.normalize(query, dim=-1)
        positive_key_norm = F.normalize(positive_key, dim=-1)
        negative_keys_norm = F.normalize(negative_keys, dim=-1)

        pos_sim = self.cosine_sim(query_norm, positive_key_norm) / self.temperature

        query_expanded = query_norm.unsqueeze(1).expand(-1, num_negatives, -1)
        neg_sim = self.cosine_sim(query_expanded, negative_keys_norm) / self.temperature

        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=query.device)

        if sample_weights is not None:
            # 使用样本权重的交叉熵损失
            loss = F.cross_entropy(all_sim, labels, reduction='none')
            weighted_loss = loss * sample_weights
            return weighted_loss.mean()
        else:
            return F.cross_entropy(all_sim, labels)