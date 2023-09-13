import os
import librosa
import torch
import torchaudio
import pandas as pd
import numpy as np

class EmodbDataset(object):
    """
        Create a Dataset for Emodb. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """
    
    _ext_audio = '.wav'
    _emotions = { 'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6 } # W = anger, L = boredom, E = disgust, A = anxiety/fear, F = happiness, T = sadness, N = neutral

    def __init__(self, root='download'):
        """
        Args:
            root (string): Directory containing the wav folder
        """
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Construct file identifiers
                    identifiers = [file[0:2], file[2:5], file[5], file[6], os.path.join('wav', file), file]

                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['speaker_id', 'code', 'emotion', 'version', 'file', 'file_name'], dtype=np.int32)

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.int32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        SAMPLE_RATE = 16000
        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, _ = librosa.load(audio_name, sr=SAMPLE_RATE)
        emotion = self.df.loc[idx, 'emotion']
        speaker_id = self.df.loc[idx, 'speaker_id']
        file_name = self.df.loc[idx, 'file_name']

        sample = {
            'waveform': waveform,
            'file_name': file_name,
            'emotion': emotion,
            'speaker_id': speaker_id
        }

        return sample

# Example: Load Emodb dataset
# emodb_dataset = EmodbDataset('/home/alanwuha/Documents/Projects/datasets/emodb/download')

# Example: Iterate through samples
# for i in range(len(emodb_dataset)):
#     sample = emodb_dataset[i]
#     print(i, sample)


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
            files += [data['file_name']]
        targets = torch.stack(targets)
        sampling_rate = 16000
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
