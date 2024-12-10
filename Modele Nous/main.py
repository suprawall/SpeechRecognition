import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import os
import DL_2024_2025_prepareData
import encoder
import sklearn.metrics as metrics

RESUME = False

def resume_training(query_encoder, key_encoder, optimizer, queue, checkpoint_path, start_epoch=0):
    # Charger le modèle et l'optimiseur depuis la sauvegarde
    checkpoint = torch.load(checkpoint_path)
    query_encoder.load_state_dict(checkpoint['model_state_dict'])
    key_encoder.load_state_dict(checkpoint['key_encoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    queue.load_state_dict(checkpoint['queue_state_dict'])
    start_epoch = checkpoint['epoch']  
    train_indices = checkpoint.get('train_indices', None)
    test_indices = checkpoint.get('test_indices', None)

    print(f"Reprise de l'entraînement à l'époque {start_epoch+1}")
    return query_encoder, key_encoder, optimizer, queue, train_indices, test_indices, start_epoch


def main():
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
    # Construire le tableau des émotions
    emotion_tab = []
    for i in range(len(target_labels)):
        emo = target_labels[i][5]
        emotion = emotion_map[emo]
        emotion_tab.append(emotion_to_idx[emotion])
        audio_file_tab.append(os.path.join(data_dir, f'{target_labels[i]}wav'))
    
    # Normalisation des spectrogrammes
    normalize = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Créer le dataset
    dataset = DL_2024_2025_prepareData.AudioDataset(audio_file_tab, emotion_tab, sr=16000, n_fft=1024, hop_length=512, transform=normalize)
    
    # Diviser le dataset en train (80%) et test (20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    query_encoder = encoder.ContrastiveModel().cuda()
    key_encoder = encoder.ContrastiveModel().cuda()
    key_encoder.load_state_dict(query_encoder.state_dict())

    # Queue pour le "momentum contrastive learning"
    queue = encoder.MemoryQueue(feature_dim=256, queue_size=1024)

    # Optimiseur
    optimizer = torch.optim.Adam(query_encoder.parameters(), lr=3e-4)

    # Fonction de perte
    criterion = nn.CrossEntropyLoss()
    
    if RESUME:
        checkpoint_path = 'model_after_24_epochs.pth'
        query_encoder, key_encoder, optimizer, queue, train_indices, test_indices, start_epoch = resume_training(
        query_encoder, key_encoder, optimizer, queue, checkpoint_path)
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    else:
        # Diviser le dataset en train (80%) et test (20%)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
        start_epoch = 0

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    


    # Fonction pour entraîner le modèle avec linear probing
    def train(model, train_loader, optimizer, start_epoch, num_epochs=10):
        model.train()
        end_epoch = start_epoch + num_epochs
        for current_epoch in range(start_epoch, end_epoch):
            print(f"Epoch {current_epoch+1}/{num_epochs}")
            running_loss = 0.0
            for i, (images, _) in enumerate(train_loader):
                print(f"iter {i}")
                """images = images.to(torch.float32)
                images = images.to('cuda')
                images = tuple(img.to(torch.float32).to('cuda') for img in images)"""
                
                spectro1, spectro2 = images
                spectro1 = spectro1.to(torch.float32).to('cuda')
                spectro2 = spectro2.to(torch.float32).to('cuda')
                
                # Query et key
                queries = query_encoder(spectro1)
                keys = key_encoder(spectro2).detach()

                # Perte contrastive
                loss = encoder.contrastive_loss_moco_v3(queries, keys, queue.queue)

                # Optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Mettre à jour l'encodeur clé
                encoder.update_key_encoder(query_encoder, key_encoder)

                # Mettre à jour la queue
                queue.enqueue_dequeue(keys)

                # Statistiques
                running_loss += loss.item()

            print(f"Train Loss: {running_loss / len(train_loader):.4f}")
            
            # Sauvegarde 
            if current_epoch + 1 == end_epoch:
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': query_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'queue_state_dict': queue.state_dict(),  
                    'key_encoder_state_dict': key_encoder.state_dict(),  
                    'train_indices': train_dataset.indices,
                    'test_indices': test_dataset.indices
                }, f'model_after_{current_epoch+1}_epochs.pth')

    # Fonction pour entraîner un classificateur linéaire sur les caractéristiques extraites
    def linear_probe(train_loader, test_loader):
        # Geler les poids du modèle pré-entraîné
        for param in query_encoder.parameters():
            param.requires_grad = False
        
        # Ajouter un classificateur linéaire (un simple MLP)
        linear_classifier = nn.Linear(256, len(set(emotion_tab))).cuda()  # Assuming 7 classes
        optimizer = optim.Adam(linear_classifier.parameters(), lr=1e-3)

        # Entraînement du classificateur linéaire
        linear_classifier.train()
        for epoch in range(20):
            print(f"Epoch {epoch+1}/{20}")
            running_loss = 0.0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(train_loader):
                print(f"iter {i}")
                print(labels)
                spectro1, spectro2 = images
                spectro1 = spectro1.to(torch.float32).to('cuda')
                spectro2 = spectro2.to(torch.float32).to('cuda')
                labels = labels.to('cuda')

                # Extraire les caractéristiques
                with torch.no_grad():
                    features = query_encoder(spectro1)
                
                # Entraîner le classificateur
                optimizer.zero_grad()
                outputs = linear_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Statistiques
                running_loss += loss.item()

                # Calculer la précision
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(f"Linear Probe Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        
        # Évaluation du classificateur linéaire sur le jeu de test
        evaluate_model(linear_classifier, test_loader)

    # Fonction pour évaluer le modèle
    def evaluate_model(model, test_loader):
        model.eval()
        all_labels = []
        all_preds = []
        print("eval...")
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                print(f"iter {i}")
                spectro1, spectro2 = images
                spectro1 = spectro1.to(torch.float32).to('cuda')
                spectro2 = spectro2.to(torch.float32).to('cuda')
                labels = labels.to('cuda')
                
                query_features = query_encoder(spectro1)  # Features pour la vue 1
                #key_features = key_encoder(spectro2)      # Features pour la vue 2

                # Utiliser uniquement query_features pour l'évaluation
                outputs = model(query_features)
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        # Calcul de la précision
        accuracy = metrics.accuracy_score(all_labels, all_preds)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")

    #checkpoint_path = 'model_after_20_epochs.pth'
    
    #model, optimizer, start_epoch = resume_training(query_encoder, optimizer, checkpoint_path, start_epoch=20)
    
    train(query_encoder, train_loader, optimizer, start_epoch=0, num_epochs=1)

    # Effectuer le linear probing (entraîner le classificateur linéaire)
    linear_probe(train_loader, test_loader)

if __name__ == '__main__':
    main()
