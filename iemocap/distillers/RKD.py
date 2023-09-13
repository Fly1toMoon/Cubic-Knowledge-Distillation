import torch
import torch.nn as nn
import torch.nn.functional as F

class RKDLoss(nn.Module):
    def __init__(self, distance_weight=0.3, angle_weight=0.7):
        super(RKDLoss, self).__init__()
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight

    @staticmethod
    def _pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def forward(self, student, teacher, target):
        stu = student.view(student.shape[0], -1)
        tea = teacher.view(teacher.shape[0], -1)

        # RKD distance loss
        with torch.no_grad():
            t_d = self._pdist(tea)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self._pdist(stu)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)

        # RKD Angle loss
        with torch.no_grad():
            td = tea.unsqueeze(0) - tea.unsqueeze(1)
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = stu.unsqueeze(0) - stu.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)

        rkd_loss = self.distance_weight * loss_d + self.angle_weight * loss_a

        students_loss = F.cross_entropy(student, target)

        total_loss = 0.5 * students_loss + 0.5 * rkd_loss

        return total_loss
