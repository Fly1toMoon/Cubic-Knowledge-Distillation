import torch
import torch.nn as nn
import torch.nn.functional as F

class PureLoss(nn.Module):
    def __init__(self):
        super(PureLoss, self).__init__()

    def forward(self, logits_student, logits_teacher, target):

        students_loss = F.cross_entropy(logits_student, target)

        return students_loss