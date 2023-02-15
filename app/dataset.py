import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys
from scipy.io.wavfile import read
from torch.utils.data import Dataset, DataLoader
import torchaudio
#import IPython.display as ipd
from sklearn.model_selection import train_test_split
import librosa
import librosa.display
from PIL import Image
import os

filename_path = 'ESC-50-Master/audio/'
#meta_df       = pd.read_csv('ESC-50-Master/meta/esc50.csv')
sample_rate   = 44100

def preprocessing(meta_df):
    
    files        = meta_df['filename'].to_numpy()
    target       = meta_df['target'].to_numpy()
    sample_rates = []
    waves        = []
    labels       = []
    #laoded .wav file converted to numpy array
    
    for count, wav_file  in enumerate(files):
        waveform, sample_rate = librosa.load(filename_path+wav_file, sr=44100)
        waves.append(waveform)
        labels.append(target[count])
        sample_rates.append(sample_rate)
    
    return waves, labels, sample_rates

def sound_identifier(meta_df):
    sound_dict = {}
    for i in meta_df.index:
        if meta_df['target'][i] not in sound_dict:
            sound_dict[meta_df['target'][i]] = meta_df['category'][i]
    return sound_dict


#waveforms, labels, sample_rates  = preprocessing(meta_df)
#X_train, X_test, y_train, y_test = train_test_split(waveforms, labels, test_size=0.20, random_state=42)
#sound_dict                       = sound_identifier(meta_df)

def create_melspectrogram_images(samples, labels, train=True):
    directory   = f'./data/melspectrogram/train/'
    
    if not train:
        directory = f'./data/melspectrogram/test/'
    
    if(os.path.isdir(directory)):
        print("Directory exists for", directory)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True),
    
    #Keep track of the images I have already saved, using counter
    existing_images = {}
    
    for i,data in enumerate(samples):
        if i % 50 == 0:
            print(i)
        count = 0
        if labels[i] in existing_images:
            existing_images[labels[i]] = existing_images[labels[i]] + 1
            count = existing_images[labels[i]]
        else:
            existing_images[labels[i]] = 0
        class_dir = sound_dict[labels[i]]
        
        filename = f'{class_dir}_spect_{count}.png'
        img_dir = os.path.join(directory, class_dir)
        img_path = os.path.join(directory, class_dir,filename)
        
        if(os.path.exists(img_path)):
            print("Skipping image")
            continue
        else:
            os.makedirs(img_dir, mode=0o777, exist_ok=True)
        
        img_path = os.path.join(img_dir, filename)
        plt.axis('off')
        X, _ = librosa.effects.trim(data)
        XS = librosa.feature.melspectrogram(X, sr=sample_rate)
        Xdb = librosa.amplitude_to_db(XS, ref=np.max)
        librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
        plt.savefig(img_path, pad_inches=0.0)

def create_mfcc_images(samples, labels, train=True):
    directory   = f'./data/mfcc/train/'
    
    if not train:
        directory = f'./data/mfcc/test/'
    
    if(os.path.isdir(directory)):
        print("Directory exists for", directory)
    else:
        os.makedirs(directory, mode=0o777, exist_ok=True),
    
    #Keep track of the images I have already saved, using counter
    existing_images = {}
    
    for i,data in enumerate(samples):
        count = 0
        if labels[i] in existing_images:
            count = existing_images[labels[i]] + 1
        else:
            existing_images[labels[i]] = 0
            
        class_dir = sound_dict[labels[i]]
        filename = f'{class_dir}_spect_{count}.png'
        img_dir = os.path.join(directory, class_dir)
        img_path = os.path.join(directory, class_dir,filename)
        if(os.path.exists(img_path)):
            print("Directory exists for", directory+class_dir)
            continue
        else:
            os.makedirs(img_dir, mode=0o777, exist_ok=True)
        

        img_path = os.path.join(img_dir, filename)
        plt.axis('off')
        mfccs = librosa.feature.mfcc(data, sample_rate)
        librosa.display.specshow(mfccs, sr=sample_rate)
        plt.savefig(img_path, pad_inches=0.0)

def create_all_images():
    #create_melspectrogram_images(X_train, y_train, train=True)
    #create_melspectrogram_images(X_test, y_test, train=False)
    #create_mfcc_images(X_train, y_train, train=True)
    #create_mfcc_images(X_test, y_test, train=False)
    pass

def load_images():
    train_melspec_path = './data/small_classes/train'
    test_melspec_path  = './data/small_classes/test'
    #train_mfcc_path    = './data/mfcc/train'
    #test_mfcc_path    = './data/mfcc/test'
    

    melspectrogram_train_dataset = datasets.ImageFolder(
        root=train_melspec_path,
        transform=transforms.Compose([transforms.ToTensor()]))
                                                       
    melspectrogram_test_dataset  = datasets.ImageFolder(
        root=test_melspec_path,
        transform=transforms.Compose([transforms.ToTensor()]))
    #mfcc_train_dataset           = datasets.ImageFolder(root=train_mfcc_path)
    #mfcc_test_dataset            = datasets.ImageFolder(root=test_mfcc_path)
    
    return melspectrogram_train_dataset, melspectrogram_test_dataset

#melspectrogram_train_dataset, melspectrogram_test_dataset = load_images()
#class_map = melspectrogram_train_dataset.class_to_idx
#TODO: write to file if file does not exist, check file later when class_map is needed