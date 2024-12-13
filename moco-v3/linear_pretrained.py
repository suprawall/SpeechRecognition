import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torchaudio
import DL_2024_2025_prepareData
import moco.builder 
from timm.models.vision_transformer import vit_base_patch16_224
from functools import partial
import vits

BATCH_SIZE = 128



# Fonction pour instancier le Vision Transformer avec une nouvelle tête
def vit_base_encoder():
    model = vit_base_patch16_224(pretrained=False)  # Charger le modèle Vision Transformer
    in_features = model.head.in_features  # Taille de sortie de la couche head
    model.head = nn.Identity()  # Supprime la tête (on ne l'utilise pas dans MoCo)
    return model

# Classe PretrainedModelWrapper
class PretrainedModelWrapper(nn.Module):
    def __init__(self, pretrained_model_path, num_classes):
        super(PretrainedModelWrapper, self).__init__()
        # Charger le modèle backbone préentraîné
        checkpoint = torch.load(pretrained_model_path, map_location="cuda")
        
        # Initialiser le backbone (base_encoder)
        self.backbone = MoCo_ViT(base_encoder=vit_base_encoder(), dim=256, mlp_dim=4096).base_encoder
        self.backbone.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Geler les poids du backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Ajouter une tête de classification
        hidden_dim = self.backbone.embed_dim  # Taille des sorties du backbone
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),  # Couche intermédiaire
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # Couche finale pour le nombre de classes
        )
    
    def forward(self, x):
        with torch.no_grad():  # Pas de gradients pour le backbone
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits



pretrained_model_path = "vit-s-300ep.pth.tar"
num_classes = 7  
model = moco.builder.MoCo_ViT(
            partial(vits.__dict__["vit_small"], stop_grad_conv1=args.stop_grad_conv1),
            256, 4096, 1.0)

# Déplacer sur GPU si disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)

# Entraîner uniquement la tête de classification
def train_model(model, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validation après chaque epoch
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100.0 * correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

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

    
segment_duration_ms = 32
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

normal_dataset = DL_2024_2025_prepareData.NormalDataset(audio_file_tab, emotion_tab, sr=16000, n_fft=512, hop_length=512, transform=normalize)

seed = 42
torch.manual_seed(seed)

# Calcul des tailles des ensembles
train_size = int(0.8 * len(normal_dataset))
test_size = len(normal_dataset) - train_size

# Mélanger les indices de manière reproductible
indices = torch.randperm(len(normal_dataset)).tolist()

# Diviser les indices en ensembles d'entraînement et de test
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Créer les sous-ensembles mélangés
train_dataset = torch.utils.data.Subset(normal_dataset, train_indices)
test_dataset = torch.utils.data.Subset(normal_dataset, test_indices)


train_sampler = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, sampler=train_sampler, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)




# Entraîner le modèle
train_model(model, train_loader, test_loader, num_epochs=10)
