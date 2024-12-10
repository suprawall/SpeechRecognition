import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        # Backbone
        self.encoder = resnet50(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, feature_dim)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.projector(features)

class MemoryQueue:
    def __init__(self, feature_dim, queue_size):
        self.queue_size = queue_size
        self.queue = torch.randn(queue_size, feature_dim).cuda()
        self.queue = F.normalize(self.queue, dim=1)  # Normalisation
        self.ptr = 0

    def enqueue_dequeue(self, keys):
        batch_size = keys.size(0)  # La taille du lot
        queue_size = self.queue.size(0)  # La taille de la queue
        
        # Vérifie si la queue peut accueillir tout le lot
        if batch_size + self.ptr > queue_size:
            # On gère le cas où on dépasse la taille de la queue
            self.ptr = 0  # Réinitialiser si nécessaire, ou faire un wrap-around
        
        # Remplir la queue avec les nouvelles clés
        self.queue[self.ptr:self.ptr + batch_size] = keys
        
        # Met à jour l'index `ptr` pour la prochaine insertion
        self.ptr = (self.ptr + batch_size) % queue_size
    def state_dict(self):
        # Retourne l'état complet de la queue (y compris les valeurs pertinentes)
        return {
            'queue': self.queue,
            'ptr': self.ptr
        }

    def load_state_dict(self, state_dict):
        # Charge l'état de la queue
        self.queue = state_dict['queue']
        self.ptr = state_dict['ptr']


def update_key_encoder(query_encoder, key_encoder, momentum=0.99):
    for param_q, param_k in zip(query_encoder.parameters(), key_encoder.parameters()):
        param_k.data = momentum * param_k.data + (1 - momentum) * param_q.data


"""def contrastive_loss(query, keys, negatives, temperature=0.07):
    query = F.normalize(query, dim=1)
    keys = F.normalize(keys, dim=1)
    negatives = F.normalize(negatives, dim=1)
    
    pos_sim = torch.sum(query * keys, dim=1) / temperature
    neg_sim = torch.mm(query, negatives.T) / temperature
    
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).cuda()  # Positive index = 0
    return F.cross_entropy(logits, labels)"""

def contrastive_loss_moco_v3(q, k, queue, temperature=0.07):
    # Normaliser les embeddings
    q = F.normalize(q, dim=1)
    k = F.normalize(k, dim=1)

    # Calculer les similarités entre les embeddings de la requête (q) et des clés (k)
    logits_pos = torch.matmul(q, k.T) / temperature  # Similarités positives (mêmes images)
    
    # Ajouter les négatifs venant de la queue
    logits_neg = torch.matmul(q, queue.T) / temperature  # Similarités négatives (vues stockées dans la queue)
    
    # Concaténer les logits positifs et négatifs
    logits = torch.cat([logits_pos, logits_neg], dim=1)

    # Nombre d'exemples dans le batch
    N = logits.shape[0]

    # Créer les labels : les vues positives ont la même étiquette
    labels = torch.arange(N).to(q.device)

    # Appliquer la CrossEntropyLoss avec les logits
    loss = F.cross_entropy(logits, labels)

    return loss
