import numpy as np
import librosa
import librosa.display
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import moco.loader

#BRUIT 
def noise(data,noise_rate=0.01):
    noise_amp = 0.01*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

#CHANGEMENT DE VITESSE
def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

#DECALAGE TEMPOREL
def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

#CHANGER LA TONALITE
def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


def augment_audio(data, augment_type, sampling_rate = 22050):
    """
    Applique les augmentations spécifiées sur un fichier audio.
    
    Args:
    - data: Le signal audio (1D NumPy array).
    - sampling_rate: La fréquence d'échantillonnage du signal.
    - augment_type: Type d'augmentation ("noise_stretch" ou "shift_pitch").
    
    Returns:
    - Le signal audio transformé.
    """
    if augment_type == "noise_stretch":
        data = noise(data)
        data = stretch(data)
    elif augment_type == "shift_pitch":
        #data = shift(data)
        data = pitch(data, sampling_rate)
    return data

class AugmentDataset(Dataset):
    def __init__(self, audio_file, emotion_tab, sr=16000, n_fft=512, hop_length=512, transform=None):
        """
        Dataset pour charger des fichiers audio et générer des spectrogrammes augmentés.
        
        Args:
        - file_paths (list): Liste des chemins vers les fichiers audio.
        - emotion_tab (list): Liste des emotions aux meme index que file_paths
        - sr (int): Fréquence d'échantillonnage (par défaut 16 kHz).
        - n_fft (int): Taille de la fenêtre FFT pour les spectrogrammes.
        - hop_length (int): Décalage entre les fenêtres FFT.
        - transform (callable, optional): Transformations supplémentaires (normalisation, etc.).
        """
        self.audio_file = audio_file
        self.emotion_tab = emotion_tab
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, idx):
        waveform = self.audio_file[idx]
        emotion = self.emotion_tab[idx]

        # Appliquer les augmentations
        augment1 = augment_audio(waveform, "noise_stretch")
        augment2 = augment_audio(waveform, "shift_pitch")

        # Convertir chaque augmentation en spectrogramme
        spectrogram1 = self._compute_spectrogram(augment1)
        spectrogram2 = self._compute_spectrogram(augment2)

        spectrogram1 = resize(torch.tensor(spectrogram1), [224, 224])
        spectrogram2 = resize(torch.tensor(spectrogram2), [224, 224])
        
        """spectrogram1 = spectrogram1.unsqueeze(0)
        spectrogram2 = spectrogram2.unsqueeze(0)"""

        # Appliquer des transformations supplémentaires si nécessaires
        if self.transform:
            spectrogram1 = self.transform(spectrogram1)
            spectrogram2 = self.transform(spectrogram2)
            
        return (spectrogram1, spectrogram2), emotion

    def _compute_spectrogram(self, data):
        """
        Convertit un signal audio en mel-spectrogramme.
        
        Args:
        - data (np.array): Signal audio brut.
        
        Returns:
        - np.array: Mel-spectrogramme 3D (3 canaux, Hauteur, Largeur).
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        mel_spectrogram_3d = np.stack([mel_spectrogram_db] * 3, axis=0)  # [3, Hauteur, Largeur]
        
        return mel_spectrogram_3d
    
class NormalDataset(Dataset):
    def __init__(self, audio_file, emotion_tab, sr=16000, n_fft=512, hop_length=512, transform=None):
        """
        Dataset pour charger des fichiers audio et générer des spectrogrammes augmentés.
        
        Args:
        - file_paths (list): Liste des chemins vers les fichiers audio.
        - emotion_tab (list): Liste des emotions aux meme index que file_paths
        - sr (int): Fréquence d'échantillonnage (par défaut 16 kHz).
        - n_fft (int): Taille de la fenêtre FFT pour les spectrogrammes.
        - hop_length (int): Décalage entre les fenêtres FFT.
        - transform (callable, optional): Transformations supplémentaires (normalisation, etc.).
        """
        self.audio_file = audio_file
        self.emotion_tab = emotion_tab
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

    def __len__(self):
        return len(self.audio_file)

    def __getitem__(self, idx):
        waveform = self.audio_file[idx]
        emotion = self.emotion_tab[idx]

        spectrogram1 = self._compute_spectrogram(waveform)

        spectrogram1 = resize(torch.tensor(spectrogram1), [224, 224])
        
        """spectrogram1 = spectrogram1.unsqueeze(0)
        spectrogram2 = spectrogram2.unsqueeze(0)"""

        # Appliquer des transformations supplémentaires si nécessaires
        if self.transform:
            spectrogram1 = self.transform(spectrogram1)
            
        return spectrogram1, emotion

    def _compute_spectrogram(self, data):
        """
        Convertit un signal audio en mel-spectrogramme.
        
        Args:
        - data (np.array): Signal audio brut.
        
        Returns:
        - np.array: Mel-spectrogramme 3D (3 canaux, Hauteur, Largeur).
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=data, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        mel_spectrogram_3d = np.stack([mel_spectrogram_db] * 3, axis=0)  # [3, Hauteur, Largeur]
        
        return mel_spectrogram_3d