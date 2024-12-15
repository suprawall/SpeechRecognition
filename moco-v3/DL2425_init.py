import os
import torchaudio
import torch
import torchvision.transforms as transforms
import DL_2425_prepareData
from torch.utils.data import DataLoader, random_split




def get_data(args):
    
    data_dir = "./wav"
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
    
    segment_duration_ms = 1024
    audio_file_tab = []
    emotion_tab = []
    for audio_file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, audio_file)
        waveform, sample_rate = torchaudio.load(file_path)
        waveform = waveform[0].numpy()
        
        speaker = audio_file[0:2]
        text_code = audio_file[2:6]
        emotion = audio_file[5]
        version = audio_file[7] if len(audio_file) > 7 else None
        label = f"{speaker}{text_code}{emotion}{version}"
        
        segment_samples = int((segment_duration_ms / 1000) * sample_rate)
        num_segments = len(waveform) // segment_samples
        
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            segment = waveform[start:end]

            # Ajouter le segment et le label correspondant
            audio_file_tab.append(segment)
            emo = emotion_map[emotion]
            emotion_tab.append(emotion_to_idx[emo])
        
        
        target_labels.append(label)
        
    normalize = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    normal_dataset = DL_2425_prepareData.NormalDataset(audio_file_tab, emotion_tab, sr=16000, n_fft=512, hop_length=512, transform=normalize)
    augment_dataset = DL_2425_prepareData.AugmentDataset(audio_file_tab, emotion_tab, sr=16000, n_fft=512, hop_length=512, transform=normalize)
    seed = 42
    torch.manual_seed(seed)

    # Calcul des tailles des ensembles
    train_size = int(0.8 * len(augment_dataset))
    test_size = len(augment_dataset) - train_size

    # Mélanger les indices de manière reproductible
    indices = torch.randperm(len(augment_dataset)).tolist()

    # Diviser les indices en ensembles d'entraînement et de test
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Créer les sous-ensembles mélangés
    train_dataset = torch.utils.data.Subset(augment_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(augment_dataset, test_indices)
    train_dataset2 = torch.utils.data.Subset(normal_dataset, train_indices)
    test_dataset2 = torch.utils.data.Subset(normal_dataset, test_indices)

    train_sampler = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    #test loader pas utile ici mais on coupe qd meme 80% pour le train loader pour pas qu'il s'entraine sur toute les données
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    train_loader2 = DataLoader(train_dataset2, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)
    test_loader2 = DataLoader(test_dataset2, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


    return train_loader, train_loader2, test_loader2