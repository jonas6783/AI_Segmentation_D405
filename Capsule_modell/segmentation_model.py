"""
Modul: segmentation_model.py
Upgrade: PointNet mit Spatial Transformer (T-Net) und Multi-Scale Features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class STN3d(nn.Module):
    """
    Spatial Transformer Network (T-Net).
    Lernt eine 3x3 Rotationsmatrix, um den Input 'geradezudrehen',
    bevor er verarbeitet wird. Macht das Netz rotationsinvariant.
    """
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9) # 3x3 Matrix = 9 Werte
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Max Pooling (Global Feature des T-Nets)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # Initialisierung als Identitätsmatrix (WICHTIG für Stabilität!)
        # Zu Beginn soll das Netz gar nichts drehen.
        iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
            
        x = x + iden
        x = x.view(-1, 3, 3) # Reshape zur 3x3 Matrix
        return x

class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSegmentation, self).__init__()
        
        # 1. Der "Ausrichter" (T-Net)
        self.stn = STN3d()
        
        # 2. Encoder (Feature Extraction)
        # Wir merken uns Outputs verschiedener Stufen (Skip Connections)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # 3. Decoder (Segmentation Head)
        # Input Size berechnet sich aus:
        # 1024 (Global) + 128 (Local High) + 64 (Local Low) = 1216 Features pro Punkt
        self.conv4 = nn.Conv1d(1216, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)
        
        self.conv6 = nn.Conv1d(256, 128, 1) # Zusätzlicher Layer für feinere Granularität
        self.bn6 = nn.BatchNorm1d(128)
        
        self.conv7 = nn.Conv1d(128, num_classes, 1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        num_points = x.size()[2]
        
        # A) T-Net Transformation
        # Wir berechnen die Matrix, die den Input "begradigt"
        trans = self.stn(x) 
        x = x.transpose(2, 1) # (B, N, 3)
        x = torch.bmm(x, trans) # Matrix-Multiplikation: Input * Rotationsmatrix
        x = x.transpose(2, 1) # Zurück zu (B, 3, N) für Convolutions
        
        # B) Encoder
        # Layer 1 (Low Level Features: Geometrie)
        out1 = F.relu(self.bn1(self.conv1(x))) # [B, 64, N]
        
        # Layer 2 (Mid Level Features: Lokale Formen)
        out2 = F.relu(self.bn2(self.conv2(out1))) # [B, 128, N]
        
        # Layer 3 (High Level: Globales Verständnis)
        out3 = F.relu(self.bn3(self.conv3(out2))) # [B, 1024, N]
        
        # Global Feature Pooling
        global_feature = torch.max(out3, 2, keepdim=True)[0] # [B, 1024, 1]
        global_feature_repeated = global_feature.repeat(1, 1, num_points)
        
        # C) Decoder mit Skip-Connections
        # Wir kleben ALLES zusammen:
        # - Was weißt du über das ganze Objekt? (Global: 1024)
        # - Was weißt du über die Form an diesem Punkt? (Mid: 128)
        # - Wo genau ist der Punkt geometrisch? (Low: 64)
        concat_feature = torch.cat([global_feature_repeated, out2, out1], 1) 
        # Gesamt: 1024 + 128 + 64 = 1216
        
        x = F.relu(self.bn4(self.conv4(concat_feature)))
        x = self.dropout(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x))) # Extra Layer
        
        # Logits
        x = self.conv7(x)
        
        return F.log_softmax(x, dim=1)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Teste Modell auf: {device}")
    
    # Test Input (Batch 2, 3 Kanäle, 2048 Punkte)
    sim_input = torch.rand(2, 3, 2048).to(device)
    model = PointNetSegmentation(num_classes=2).to(device)
    
    output = model(sim_input)
    print("Output Shape:", output.shape) # Sollte [2, 2, 2048] sein
    
    # Kurzer Check ob T-Net läuft
    try:
        dummy_trans = model.stn(sim_input)
        print("T-Net Matrix Shape:", dummy_trans.shape) # [2, 3, 3]
        print(">> Modell Architektur erfolgreich verifiziert.")
    except Exception as e:
        print(">> Fehler im T-Net:", e)
