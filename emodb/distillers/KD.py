import torch
import torch.nn as nn
import torch.nn.functional as F

class KDLoss(nn.Module):
    def __init__(self, temp=1.2, alpha=0.7):
        super(KDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha

    def forward(self, logits_student, logits_teacher, target):

        students_loss = F.cross_entropy(logits_student, target)
        ditillation_loss = F.kl_div(
            F.log_softmax(logits_student / self.temp, dim=1),
            F.softmax(logits_teacher / self.temp, dim=1)
        )
        kd_loss = ditillation_loss * (self.alpha * self.temp * self.temp) + students_loss * (1 - self.alpha) 

        return kd_loss