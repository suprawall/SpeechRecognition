import numpy as np
import librosa
import librosa.display
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize
import torchvision.transforms as transforms

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

################################################################
#
#adapter l'augmentation des données (par paire d'audio original)
#
################################################################
def augment_audio(data, sampling_rate, augment_type):
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
        data = shift(data)
        data = pitch(data, sampling_rate)
    return data

class AudioDataset(Dataset):
    def __init__(self, file_paths, sr=16000, n_fft=1024, hop_length=512, transform=None):
        """
        Dataset pour charger des fichiers audio et générer des spectrogrammes augmentés.
        
        Args:
        - file_paths (list): Liste des chemins vers les fichiers audio.
        - sr (int): Fréquence d'échantillonnage (par défaut 16 kHz).
        - n_fft (int): Taille de la fenêtre FFT pour les spectrogrammes.
        - hop_length (int): Décalage entre les fenêtres FFT.
        - transform (callable, optional): Transformations supplémentaires (normalisation, etc.).
        """
        self.file_paths = file_paths
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Charger le fichier audio
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        waveform = waveform[0].numpy()  # Convertir en NumPy array

        # Appliquer les augmentations
        augment1 = augment_audio(waveform, sr, "noise_stretch")
        augment2 = augment_audio(waveform, sr, "shift_pitch")

        # Convertir chaque augmentation en spectrogramme
        spectrogram1 = self._compute_spectrogram(augment1, sr)
        spectrogram2 = self._compute_spectrogram(augment2, sr)
    
        
        spectrogram1 = resize(torch.tensor(spectrogram1), [224, 224])
        spectrogram2 = resize(torch.tensor(spectrogram2), [224, 224])
        
        spectrogram1 = spectrogram1.unsqueeze(0)
        spectrogram2 = spectrogram2.unsqueeze(0)

        # Appliquer des transformations supplémentaires si nécessaires
        if self.transform:
            spectrogram1 = self.transform(spectrogram1)
            spectrogram2 = self.transform(spectrogram2)
            
        return spectrogram1, spectrogram2

    def _compute_spectrogram(self, data, sr):
        """
        Convertit un signal audio en spectrogramme.
        
        Args:
        - data (np.array): Signal audio brut.
        - sr (int): Fréquence d'échantillonnage.
        
        Returns:
        - np.array: Spectrogramme 3D (3 canaux, Hauteur, Largeur).
        """
        # Calcul du spectrogramme
        spectrogram = librosa.stft(data, n_fft=self.n_fft, hop_length=self.hop_length)
        spectrogram = np.abs(spectrogram)  
        spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
        spectrogram_3d = np.stack([spectrogram_db] * 3, axis=0)  # [3, Hauteur, Largeur]
          
        return spectrogram_3d

