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

from dataset import EmodbDataset, Collator
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
ap.add_argument("-input", "--input_type",  
                default="mel",
                type=str,
                help="input_type")

ap.add_argument("-model", "--stu_model",  
                default="mobilenetv3_small_075",
                type=str,
                help="name of student model")

ap.add_argument("-kd", "--kd_type",  
                default="pure",
                type=str,
                help="kd_type")

ap.add_argument("-lr", "--learning_rate",
                default=0.001,
                type=float,
                help="learning rate")

ap.add_argument("-bt", "--batch_size",
                default=16,
                type=int,
                help="batch_size")

ap.add_argument("-epoc", "--epoc_nums",
                default=100,
                type=int,
                help="epoc_nums")

ap.add_argument("-l1", "--lambda_1",
                default=0.006,
                type=float,
                help="lambda1")

ap.add_argument("-l2", "--lambda_2",
                default=0.015,
                type=float,
                help="lambda2")

ap.add_argument("-len", "--input_length",
                default=80000,
                type=int,
                help="input_length")

args = vars(ap.parse_args())

input_type = args["input_type"]
stu_model = args["stu_model"]
kd_type = args["kd_type"]
lr = args["learning_rate"]
batch_size = args["batch_size"]
epoc_nums = args["epoc_nums"]
lambda_1 = args["lambda_1"]
lambda_2 = args["lambda_2"]
input_length = args["input_length"]


sample_rate = 16000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = epoc_nums
learning_rate = lr
num_labels = 7


# Your data_dir here
data_dir = ''
dataset = EmodbDataset(root=data_dir)

extractor = Wav2Vec2FeatureExtractor.from_pretrained('superb/wav2vec2-base-superb-er') # for ER
teacher_predictions_by_path = torch.load("./tea_output/tea_preds_by_path.pth")

train_collator = Collator(extractor, input_length, teacher_predictions_by_path, input_type) 
test_collator = Collator(extractor, input_length, teacher_predictions_by_path, input_type)

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
        self.classifier = nn.Linear(128, 7)
        
    def forward(self, inputs):
        features = self.efficientnet(inputs)
        outputs = self.classifier(features)
        return outputs

def train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, fold, log_interval=10):
    print(f"Fold : {fold}")
    torch.backends.cudnn.benchmark = True

    UA = 0.0
    WA = 0.0

    log_intervals = 10
    pbar_update = 1 / sum([len(v) for v in dataloaders_dict.values()])

    n_step = 0
    with tqdm(total=num_epochs) as pbar:
        val_wa = [] # weighted accuracy
        val_ua = [] # unweighted accuracy
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                epoch_loss = 0.0
                epoch_corrects = 0
                class_corrects = np.zeros(num_labels)
                target_counts = np.zeros(num_labels)

                
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
                            n_step += len(stu_inputs)
                            loss.backward()
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                            loss_log = loss.item()
                            del loss
                            if step % log_interval == 0:
                                print(
                                    f"Train Epoch: {epoch} [{step * len(stu_inputs)}/{len(dataloaders_dict[phase].dataset)} ({100. * step / len(dataloaders_dict[phase]):.0f}%)]\tLoss: {loss_log:.6f}")

                        else:
                            for p, t in zip(preds, target):
                                target_counts[t] += 1
                                if p == t:
                                    class_corrects[t] += 1

                        epoch_loss += loss_log * len(stu_inputs)
                        epoch_corrects += preds.squeeze().eq(target).sum().item()

                        pbar.update(pbar_update)


                if phase == 'train':
                    len_data = len(train_dataset)
                else:
                    len_data = len(test_dataset)

                epoch_loss = epoch_loss / len_data
                epoch_acc = epoch_corrects / len_data

                if phase == 'train':
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch + 1, num_epochs, phase, epoch_loss, epoch_acc))
                
                else:
                    val_ua.append(epoch_acc)
                    epoch_uacc = 0.
                    class_count = 0
                    for i in range(len(target_counts)):
                        if target_counts[i] != 0:
                            # Avoid a denominator of 0
                            epoch_uacc += class_corrects[i] / target_counts[i]
                            class_count += 1
                    epoch_uacc = epoch_uacc / class_count
                    val_wa.append(epoch_uacc)
                    if epoch_acc > WA:
                        WA = epoch_acc
                        UA = epoch_uacc
                    print('Epoch {}/{} | {:^5} |  Loss: {:.4f} UA: {:.4f} WA: {:.4f}'.format(epoch+1, num_epochs, phase, epoch_loss, epoch_acc, epoch_uacc))

    print("UA:", UA)
    print("WA:", WA)
    UAs[fold] = UA
    WAs[fold] = WA
    return model


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        return sample


list_of_speaker_ids = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]

# Define the number of folds (10-fold cross validation)
num_folds = 10

UAs = np.zeros(num_folds)
WAs = np.zeros(num_folds)
best_epoc = np.zeros(num_folds)


kf = KFold(n_splits=num_folds, shuffle=False)
fold = 0

print('*'*40)
print(f"The student model is {stu_model}")
print('*'*40)

for train_indices, test_indices in kf.split(list_of_speaker_ids):
    train_speaker_ids = [list_of_speaker_ids[i] for i in train_indices]
    test_speaker_ids = [list_of_speaker_ids[i] for i in test_indices]
    
    # Split the dataset based on train_speaker_ids and test_speaker_ids
    train_data = [dataset[i] for i in range(len(dataset)) if dataset[i]['speaker_id'] in train_speaker_ids]
    test_data = [dataset[i] for i in range(len(dataset)) if dataset[i]['speaker_id'] in test_speaker_ids]

    train_dataset = CustomDataset(train_data)
    test_dataset = CustomDataset(test_data)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_collator, shuffle = True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=test_collator, shuffle = False, num_workers=8, pin_memory=True)
    dataloaders_dict = {'train': train_loader, 'val': test_loader}

    model = SERcnn(stu_model)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        params = model.module.parameters()
    else:
        params = model.parameters()

    model.to(device)

    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.9)

    model = train_model(model, dataloaders_dict, optimizer, scheduler, num_epochs, fold, log_interval=10)
    fold += 1


print(f"The mean of UA is:{UAs.mean()} , the std of UA is:{UAs.std()}")
print(f"The mean of WA is:{WAs.mean()} , the std of WA is:{WAs.std()}")

