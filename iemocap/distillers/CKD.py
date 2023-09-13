import torch
import torch.nn as nn
import torch.nn.functional as F

class CKDLoss(nn.Module):
    def __init__(self, temp_range=6, lambda_1=1.0, lambda_2=1.0, alpha=0.7):
        super(CKDLoss, self).__init__()
        self.temp_range = temp_range
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha = alpha

    def forward(self, logits_student, logits_teacher, target):

        loss_kd = self.get_kdloss(logits_student, logits_teacher, target)

        student_cube = self.get_cube(logits_student)
        teacher_cube = self.get_cube(logits_teacher)

        stu_scd = self.get_scd(student_cube)
        tea_scd = self.get_scd(teacher_cube)
        scd = tea_scd - stu_scd

        loss_l1 = 0.00025 * F.l1_loss(stu_scd, tea_scd, reduction='sum')
        loss_l2 = 0.00025 * F.mse_loss(stu_scd, tea_scd, reduction='sum')
        
        loss_sub = self.get_subloss(scd, self.lambda_1, self.lambda_2)

        loss = loss_kd + loss_l1 + loss_l2 + loss_sub 

        return loss

    def get_kdloss(self, logits_student, logits_teacher, target):
        loss = 0.
        for temp in range(1, self.temp_range):
            students_loss = F.cross_entropy(logits_student, target)

            ditillation_loss = F.kl_div(
                F.log_softmax(logits_student / temp, dim=1),
                F.softmax(logits_teacher / temp, dim=1)
            )
            kd_loss = ditillation_loss * (self.alpha * temp * temp) + students_loss * (1 - self.alpha) 
            loss += kd_loss
        return loss

    def get_cube(self, logits):
        cube = []
        for temp in range(1, self.temp_range):
            cube.append(F.softmax(logits / temp, dim=1))
        cube = torch.stack(cube, dim=-1)
        return cube

    def get_scd(cube):
        scd = []
        for i in range(cube.shape[0]):
            line_j = []
            for j in range(cube.shape[1]):
                line_k = []
                for k in range(cube.shape[2]):
                    line_k.append(cube[i,j,k] * cube)
                line_j.append(torch.stack(line_k, dim=0))
            scd.append(torch.stack(line_j, dim=0))
        scd = torch.stack(scd, dim=0)
        return scd

    def get_subloss(scd, lamb1=0.001, lamb2=0.015):
        loss_sub1 = 0.
        loss_sub2 = 0.
        count = 0
        for k in range(scd.shape[2]):
            for i in range(scd.shape[0]):
                for l in range(scd.shape[0]): 
                    for j in range(scd.shape[1]):
                        for m in range(scd.shape[1]):
                            loss_sub1 += scd[l,j,k,i,j,k] * scd[l,m,k,i,m,k]
                            loss_sub2 += scd[i,m,k,i,j,k] * scd[l,m,k,l,j,k]
        return lamb1 * loss_sub1 + lamb2 * loss_sub2
