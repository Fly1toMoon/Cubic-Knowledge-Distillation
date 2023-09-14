# Cubic-Knowledge-Distillation
Code for 'Cubic Knowledge Distillation for Speech Emotion Recognition'

# Abstract
Speech Emotion Recognition (SER) can play an important role in human-computer interaction. In this paper, we propose a logit knowledge distillation method for SER, called Cubic KD, that distill the knowledge of fine-tuned self-supervised models to allow better performance of small models. By creating cubic structures from teacher and student network output features and using a loss function to distill the cube structure through self-correlation between elements, Cubic KD efficiently captures knowledge within instances and among instances. We apply this distillation method to four student models and conduct experiments using the Emo-DB and IEMOCAP datasets. The results show that Cubic KD outperforms existing predictive logit knowledge distillation methods and is comparable to intermediate feature knowledge distillation methods.

![figkd_page-0001-2](https://github.com/Fly1toMoon/Cubic-Knowledge-Distillation/assets/87889196/6f6730d0-5cce-489b-8476-d67fa6721e2a)

# Installation and Training
### 1. Requirements:
You need to install packages necessary for running experiments. Please run the following command.
```sh
pip install -r requirements.txt
```
### 2. Data:
Download **[EMO-DB](http://emodb.bilderbar.info/download/download.zip)** and **[IEMOCAP](https://sail.usc.edu/iemocap/iemocap_release.htm)**(requires permission to access) datasets.

### 3.Training:
If you want to train on Emo-DB dataset:
```sh
cd emodb
```
Or you want to train on Emo-DB dataset:
```sh
cd iemocap
```
Then you can use the following command to train Light-SERNet+ model:
```sh
python train_light.py -session {test session number}(only on IEMOCAP) \
                      -kd {KD mothod(pure, KD, RKD, DKD, CKD)} \
                      -input {type of input(mfcc, mel_spectrogram)} \
                      -lr {learning_rate} \
                      -bt {batch_size} \
                      -epoc {epoc_nums} \
                      -l1 {lambda1 for Cubic KD} \
                      -l2 {lambda2 for Cubic KD} \
                      -len {input length for audio}
```
You can also train other models:
```sh
python train_cnn.py -session {test session number}(only on IEMOCAP) \
                    -model {name of model on timm} \
                    -kd {KD mothod(pure, KD, RKD, DKD, CKD)} \
                    -input {type of input(mfcc, mel_spectrogram)} \
                    -lr {learning_rate} \
                    -bt {batch_size} \
                    -epoc {epoc_nums} \
                    -l1 {lambda1 for Cubic KD} \
                    -l2 {lambda2 for Cubic KD} \
                    -len {input length for audio}
```
#### Example:
You can use following command to train Mobilenet model on IEMOCAP using the CubicKD method with session1 as test set.
```sh
cd iemocap
python train_cnn.py -session 1 \
                    -model "mobilenetv3_small_075" \
                    -kd "CKD" \
                    -input "mel" \
                    -lr 0.0001 \
                    -bt 32 \
                    -epoc 200 \
                    -l1 0.001 \
                    -l2 0.015 \
                    -len 200000
```

# Results:
We use Light-SERNet+, MobileNetV3, EfficientNetB0, ResNet18 as student models, WavLM Base+ as teacher model, and the experimental results on Emo-DB and IEMOCAP datasets are as follows：
<img width="1167" alt="截屏2023-09-14 17 28 21" src="https://github.com/Fly1toMoon/Cubic-Knowledge-Distillation/assets/87889196/4236b289-fa0e-41a8-b5c7-412e49e86ee0">
