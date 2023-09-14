import torch, torchaudio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import random
import timm
import argparse

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from statistics import mean
from torch.utils.data import random_split, DataLoader, SubsetRandomSampler, Dataset
from transformers import Wav2Vec2FeatureExtractor 
from tqdm import tqdm

from dataset import IEMOCAPDataset, Collator
from distillers.CKD import CKDLoss
from distillers.DKD import DKDLoss
from distillers.RKD import RKDLoss
from distillers.KD import KDLoss
from distillers.pure import PureLoss


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

torch_fix_seed()
# utils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-session", "--test_session",
                default=1,
                type=int,
                help="test_session")

ap.add_argument("-model", "--stu_model",  
                default="mobilenetv3_small_075",
                type=str,
                help="name of student model")

ap.add_argument("-kd", "--kd_type",  
                default="pure",
                type=str,
                help="kd_type")

ap.add_argument("-input", "--input_type",  
                default="mel",
                type=str,
                help="input_type")

ap.add_argument("-lr", "--learning_rate",
                default=0.0001,
                type=float,
                help="learning rate")

ap.add_argument("-bt", "--batch_size",
                default=32,
                type=int,
                help="batch_size")

ap.add_argument("-epoc", "--epoc_nums",
                default=50,
                type=int,
                help="epoc_nums")

ap.add_argument("-l1", "--lambda_1",
                default=0.001,
                type=float,
                help="lambda_l1")

ap.add_argument("-l2", "--lambda_2",
                default=0.015,
                type=float,
                help="lambda_l2")

ap.add_argument("-len", "--input_length",
                default=200000,
                type=int,
                help="input_length")

args = vars(ap.parse_args())

test_session = args["test_session"]
stu_model = args["stu_model"]
kd_type = args["kd_type"]
input_type = args["input_type"]
lr = args["learning_rate"]
batch_size = args["batch_size"]
epoc_nums = args["epoc_nums"]
lambda_1 = args["lambda_1"]
lambda_2 = args["lambda_2"]
input_length = args["input_length"]


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

sample_rate = 16000
num_epochs = epoc_nums
learning_rate = lr
pretrained_model = "microsoft/wavlm-base-plus"
num_labels = 4

extractor = Wav2Vec2FeatureExtractor.from_pretrained('superb/wav2vec2-base-superb-er') # for ER
teacher_predictions_by_path = torch.load("./tea_output/tea_preds_by_path.pth")

# Your IEMOCAP dataset path here
data_path = ''

train_dataset = IEMOCAPDataset(root=data_path, fold_num=test_session, split='train')
test_dataset = IEMOCAPDataset(root=data_path, fold_num=test_session, split='test')

print('*'*50)
print("test_session : ", test_session)
print('*'*50)

train_collator = Collator(extractor, input_length, teacher_predictions_by_path, input_type)
test_collator = Collator(extractor, input_length, teacher_predictions_by_path, input_type)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_collator, num_workers=8, pin_memory=True)
dataloaders_dict = {'train': train_loader, 'test': test_loader}

if kd_type == "pure":
    loss_func = PureLoss()
elif kd_type == "KD":
    loss_func = KDLoss()
elif kd_type == "RKD":
    loss_func = RKDLoss()
elif kd_type == "DKD":
    loss_func = DKDLoss()
elif kd_type == "CKD":
    loss_func = CKDLoss(lambda_1=lambda_1, lambda_2=lambda_2)


class SERcnn(nn.Module):
    def __init__(self, stu_model):
        super().__init__()
        self.model_name = stu_model
        self.efficientnet = timm.create_model(self.model_name, pretrained=False, in_chans=1, num_classes=128)
        self.classifier = nn.Linear(128, 4)
        
    def forward(self, inputs):
        features = self.efficientnet(inputs)
        outputs = self.classifier(features)
        return outputs


def train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, log_interval=10):

    UA = 0.
    WA = 0.
    torch.backends.cudnn.benchmark = True

    log_intervals = 10
    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])

    n_step = 0
    with tqdm(total=num_epochs) as pbar:
        val_wa = [] # weighted accuracy
        val_ua = [] # unweighted accuracy
        best_ua = 0.0
        best_wa = 0.0
        for epoch in range(num_epochs):
            for phase in ['train', 'test']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                epoch_loss = 0.0
                epoch_corrects = 0
                class_corrects = np.zeros(num_labels)
                target_counts = np.zeros(num_labels)
                count = 1
                
                for step, (data, target, teachers_preds) in enumerate(dataloaders_dict[phase]):

                    target = target.to(device)

                    teachers_preds = teachers_preds.to(device)

                    stu_inputs = data.to(device)
                    stu_inputs = stu_inputs.unsqueeze(1)
                   

                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                                            
                        students_preds = model(stu_inputs)

                        loss = loss_func(students_preds, teachers_preds, target)

                        preds = torch.argmax(students_preds, dim=-1) 

                        if phase == 'train':
                            n_step += len(data)
                            loss.backward()
                            optimizer.step()
                            scheduler.step()

                            loss_log = loss.item()
                            del loss
                            if step % log_interval == 0:
                                print(
                                    f"Train Epoch: {epoch} [{step * len(data)}/{len(dataloaders_dict[phase].dataset)} ({100. * step / len(dataloaders_dict[phase]):.0f}%)]\tLoss: {loss_log:.6f}")

                        else:
                            for p, t in zip(preds, target):
                                target_counts[t] += 1
                                if p == t:
                                    class_corrects[t] += 1

                        epoch_loss += loss_log * len(data)
                        epoch_corrects += preds.squeeze().eq(target).sum().item()

                        pbar.update(pbar_update)

                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects / len(dataloaders_dict[phase].dataset)

                if phase == 'train':
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))

                else:
                    val_ua.append(epoch_acc)
                    epoch_uacc = (class_corrects / target_counts).mean()
                    val_wa.append(epoch_uacc)
                    if epoch_acc > WA:
                        WA = epoch_acc
                        UA = epoch_uacc
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} WA: {:.4f} UA: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc, epoch_uacc))

    print(f"The WA of session{test_session} is {WA}")
    print(f"The UA of session{test_session} is {UA}")
    return model


model = SERcnn(stu_model)


if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    params = model.module.parameters()
else:
    params = model.parameters()

model.to(device)

optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)


model = train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, log_interval=10)


