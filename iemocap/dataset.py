from pathlib import Path
import os
import librosa
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

SAMPLE_RATE = 16000

metadata_dir = Path(
    './s3prl_split')



class IEMOCAPDataset(Dataset):
    def __init__(self, root='', fold_num=1, split='train', pre_load=True):
        metadata_path = metadata_dir / \
            f'Session{fold_num}' / f'{split}_meta_data.csv'
        self.root = root
        self.df = pd.read_csv(metadata_path)
        self.class_num = len(np.unique(self.df['label'].values))

    def __getitem__(self, idx):
        data = self.df.loc[idx]
        audio_path = os.path.join(self.root, data['path'])
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        return {'waveform': waveform,
                'emotion': data['label'],
                'path': data['path']}

    def __len__(self):
        return len(self.df)



# get MFCC
def feature_spectrogram(
    waveform, 
    fft = 1024,
    hop=256,
    ):
    
    spectrogram = librosa.stft(
        y=waveform, 
        n_fft=fft, 
        hop_length=hop
        )
    
    spectrogram = np.abs(spectrogram)  # 取幅值
    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # 转换为对数刻度
    
    return spectrogram

def feature_melspectrogram(
    waveform, 
    sample_rate = 16000,
    fft = 1024,
    hop=256,
    mels=128,
    ):
    
    # Produce the mel spectrogram for all STFT frames and get the mean of each column of the resulting matrix to create a feature array
    # Using 8khz as upper frequency bound should be enough for most speech classification tasks
    melspectrogram = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sample_rate, 
        n_fft=fft, 
        hop_length=hop, 
        n_mels=mels
        )
    
    # convert from power (amplitude**2) to decibels
    # necessary for network to learn - doesn't converge with raw power spectrograms 
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    
    return melspectrogram

def feature_mfcc(
    waveform, 
    sample_rate = 16000,
    n_mfcc = 40,
    fft = 1024,
    winlen = 512,
    window='hamming',
    #hop=256, # increases # of time steps; was not helpful
    mels=128
    ):

    # Compute the MFCCs for all STFT frames 
    # 40 mel filterbanks (n_mfcc) = 40 coefficients
    mfc_coefficients=librosa.feature.mfcc(
        y=waveform, 
        sr=sample_rate, 
        n_mfcc=n_mfcc,
        n_fft=fft, 
        win_length=winlen, 
        window=window, 
        #hop_length=hop, 
        n_mels=mels, 
        fmax=sample_rate/2
        ) 

    return mfc_coefficients



class Collator:
    def __init__(self, extractor=None, lthresh=None, teacher_predictions_by_path=None, input_type="mel"):
        self.extractor = extractor
        self.lthresh = lthresh
        self.teacher_predictions_by_path = teacher_predictions_by_path
        self.input_type = input_type

    def __call__(self, batch):
        waveforms, targets = [], []
        files = []
        for data in batch:
            if self.lthresh is None:
                waveforms += [data['waveform'].flatten()]
            else:
                total_len = len(data['waveform'])
                if total_len > self.lthresh:
                    start_index = (total_len - self.lthresh) // 2
                    waveforms += [data['waveform'].flatten()[start_index:start_index + self.lthresh]]
                else:
                    waveforms += [data['waveform'].flatten()[:self.lthresh]]
            targets += [torch.tensor(int(data['emotion']))]
            files += [data['path']]
        targets = torch.stack(targets)
        sampling_rate = self.extractor.sampling_rate
        inputs = self.extractor(
            waveforms,
            sampling_rate=sampling_rate,
            padding='max_length',
            max_length=self.lthresh,
            return_tensors='pt')

        if self.input_type == "mfcc":
            stu_inputs = torch.tensor(feature_mfcc(inputs['input_values'].numpy()))
        else:
            stu_inputs = torch.tensor(feature_melspectrogram(inputs['input_values'].numpy()))
        
        teachers_preds = [self.teacher_predictions_by_path[files[i]] for i in range(len(files))]
        teachers_preds = torch.stack(teachers_preds)

        sample = (stu_inputs, targets, teachers_preds)

        return sample


if __name__ == '__main__':
    train_dataset = IEMOCAPDataset(fold_num=1, split='train')
    print(train_dataset[0])
    test_dataset = IEMOCAPDataset(fold_num=1, split='test')
    print(test_dataset[0])