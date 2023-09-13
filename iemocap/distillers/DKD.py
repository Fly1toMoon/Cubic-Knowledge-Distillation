import torch
import torch.nn as nn
import torch.nn.functional as F

class DKDLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.25, temperature=1):
        super(DKDLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits_student, logits_teacher, target):
        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / self.temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
            * (self.temperature**2)
            / target.shape[0]
        )
        return self.alpha * tckd_loss + self.beta * nckd_loss

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdim=True)
        t2 = (t * mask2).sum(1, keepdim=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt
