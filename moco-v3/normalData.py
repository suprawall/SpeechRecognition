import DL_2024_2025_prepareData
import os
import torchvision.transforms as transforms
from torch.utils.data import random_split, Subset

# Data loading code
data_dir = "./wav"

# Load and preprocess audio data using spectrograms
labels = os.listdir(data_dir)
audio_data = []
target_labels = []

emotion_map = {
    'W': 'anger',
    'L': 'boredom',
    'E': 'disgust',
    'A': 'fear',
    'F': 'happiness',
    'N': 'neutral',
    'T': 'sadness'
}

emotion_to_idx = {
    'anger': 0,
    'boredom': 1,
    'disgust': 2,
    'fear': 3,
    'happiness': 4,
    'neutral': 5,
    'sadness': 6
}

for audio_file in os.listdir(data_dir):
    speaker = audio_file[0:2]
    text_code = audio_file[2:6]
    emotion = audio_file[6]
    version = audio_file[7] if len(audio_file) > 7 else None
    label = f"{speaker}{text_code}{emotion}{version}"
    target_labels.append(label)

audio_file_tab = []
#construire le tableau des emotions:
emotion_tab = []
for i in range(len(target_labels)):
    emo = target_labels[i][5]
    emotion = emotion_map[emo]
    emotion_tab.append(emotion_to_idx[emotion])
    audio_file_tab.append(os.path.join(data_dir, f'{target_labels[i]}wav'))
emotion_tab
    
normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

normal_dataset = DL_2024_2025_prepareData.NormalDataset(audio_file_tab, emotion_tab, sr=16000, n_fft=1024, hop_length=512, transform=normalize)

train_size = int(0.8 * len(normal_dataset))
test_size = len(normal_dataset) - train_size

# Diviser l'index de manière fixe, sans randomisation
train_indices = list(range(train_size))
test_indices = list(range(train_size, len(normal_dataset)))

train_dataset = Subset(normal_dataset, train_indices)
test_dataset = Subset(normal_dataset, test_indices)