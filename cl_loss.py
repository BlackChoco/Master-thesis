import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    对比损失函数，支持三种InfoNCE变体
    """
    def __init__(self, temperature: float = 0.07, loss_type: str = 'infonce', 
                 infonce_mode: str = 'unidirectional'):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.infonce_mode = infonce_mode
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        print(f"🎯 ContrastiveLoss配置: 类型={loss_type}, InfoNCE模式={infonce_mode}, 温度={temperature}")
        
    def forward(self, anchor, positive, negatives=None):
        if anchor.nelement() == 0 or positive.nelement() == 0:
            return torch.tensor(0.0, device=anchor.device if anchor.nelement() > 0 else positive.device, requires_grad=True)

        if self.loss_type == 'infonce':
            if self.infonce_mode == 'unidirectional':
                return self._infonce_loss_unidirectional(anchor, positive, negatives)
            elif self.infonce_mode == 'bidirectional':
                return self._infonce_loss_bidirectional(anchor, positive, negatives)
            elif self.infonce_mode == 'in_batch':
                return self._infonce_loss_in_batch(anchor, positive)
            else:
                raise ValueError(f"不支持的 InfoNCE 模式: {self.infonce_mode}")
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
    
    def _infonce_loss_unidirectional(self, anchor, positive, negatives):
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
        loss = F.cross_entropy(all_sim, labels)
        return loss
    
    def _infonce_loss_bidirectional(self, anchor, positive, negatives):
        if negatives is None or negatives.nelement() == 0:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
            
        loss1 = self._compute_single_direction_loss(anchor, positive, negatives)
        loss2 = self._compute_single_direction_loss(positive, anchor, negatives)
        return (loss1 + loss2) / 2.0
    
    def _infonce_loss_in_batch(self, anchor, positive):
        batch_size = anchor.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=anchor.device, requires_grad=True)
        
        anchor_norm = F.normalize(anchor, dim=-1)
        positive_norm = F.normalize(positive, dim=-1)
        
        similarity_matrix_ap = torch.matmul(anchor_norm, positive_norm.t()) / self.temperature
        labels_ap = torch.arange(batch_size, device=anchor.device)
        loss_ap = F.cross_entropy(similarity_matrix_ap, labels_ap)
        
        similarity_matrix_pa = torch.matmul(positive_norm, anchor_norm.t()) / self.temperature
        labels_pa = torch.arange(batch_size, device=anchor.device)
        loss_pa = F.cross_entropy(similarity_matrix_pa, labels_pa)

        return (loss_ap + loss_pa) / 2.0

    def _compute_single_direction_loss(self, query, positive_key, negative_keys):
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
        loss = F.cross_entropy(all_sim, labels)
        return loss