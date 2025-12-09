import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import time

# Dein Modell importieren
from segmentation_model import PointNetSegmentation

# ==========================================
#   KONFIGURATION
# ==========================================
CONFIG = {
    "data_dir": "training_data",
    "model_save_path": "best_segmentation_model.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # System
    "num_workers": 4,         
    
    # Training
    "batch_size": 16,         # 32 kann bei 2048 Punkten + Gradients viel VRAM fressen. 16 ist sicherer.
    "epochs": 80,             # Durch Augmentation brauchen wir evtl. länger, aber 80 ist gut.
    "learning_rate": 0.001,   
    
    # Gewichtung der Klassen (WICHTIG!)
    # Hintergrund (0) vs. Bauteil (1). 
    # Da wir nun Tischplatten haben, ist das Verhältnis oft unausgewogen.
    # Wir strafen Fehler am Bauteil (1.0) meist härter ab als am Hintergrund, 
    # oder wir balancieren es. Hier ein Startwert:
    "class_weights": [1.0, 1.5], # [Background, Object] -> Objekt ist wichtiger!
}

# ==========================================
#   HELPER: Data Augmentation
# ==========================================
def augment_batch(points):
    """
    Wendet zufällige Transformationen auf die Punktwolke an (während des Trainings).
    points: (3, N)
    """
    # 1. Leichte Rotation um Y-Achse (oft die "Oben"-Achse)
    theta = np.random.uniform(0, 2*np.pi)
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    
    # Transponieren für Matrix-Mult: (3,N) -> (N,3)
    points = points.T 
    points = np.dot(points, rotation_matrix)
    
    # 2. Skalierung (Bauteile wirken je nach Kameraabstand größer/kleiner)
    scale = np.random.uniform(0.9, 1.1)
    points *= scale
    
    # 3. Jitter (Simuliert Sensorrauschen)
    jitter = np.random.normal(0, 0.005, points.shape)
    points += jitter
    
    # Zurück zu (3, N)
    return points.T.astype(np.float32)

# ==========================================
#   DATASET
# ==========================================
class PointCloudDataset(Dataset):
    def __init__(self, points_file, labels_file, augment=False):
        self.points = np.load(points_file).astype(np.float32)
        self.labels = np.load(labels_file).astype(np.int64)
        self.augment = augment # Nur im Training True setzen!
        
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        points = self.points[idx]      # (N, 3)
        target = self.labels[idx]      # (N)
        
        # Transponieren für PointNet: (N, 3) -> (3, N)
        points = points.transpose(1, 0)
        
        if self.augment:
            points = augment_batch(points)
            
        return points, target

# ==========================================
#   METRIKEN
# ==========================================
def calculate_metrics(pred, target, num_classes=2):
    """
    Berechnet Accuracy und IoU speziell für das Bauteil (Klasse 1).
    """
    pred_choice = pred.data.max(1)[1] # (B, N)
    
    # 1. Accuracy (Global)
    correct = pred_choice.eq(target.data).cpu().sum()
    total = target.numel()
    accuracy = correct.item() / float(total)
    
    # 2. IoU für Klasse 1 (Objekt) - Das ist die wichtigste Zahl!
    # Wir wollen wissen: Wie gut wurde das Bauteil getroffen?
    intersection = (pred_choice == 1) & (target == 1)
    union = (pred_choice == 1) | (target == 1)
    
    intersection_sum = intersection.sum().float()
    union_sum = union.sum().float()
    
    if union_sum == 0:
        iou_obj = 1.0 # Perfekt, wenn es kein Objekt gab und keins vorhergesagt wurde
    else:
        iou_obj = intersection_sum / union_sum
        
    return accuracy, iou_obj.item()

# ==========================================
#   TRAINING LOOP
# ==========================================
def train():
    print(f"--- Starte Training auf: {CONFIG['device']} ---")
    
    # 1. Daten laden
    try:
        points_path = os.path.join(CONFIG["data_dir"], "train_points.npy")
        labels_path = os.path.join(CONFIG["data_dir"], "train_labels.npy")
        
        # Basis-Dataset laden
        full_dataset = PointCloudDataset(points_path, labels_path, augment=False)
    except FileNotFoundError:
        print("CRITICAL: Daten nicht gefunden. Erst data_generator.py ausführen!")
        return

    # 2. Split in Train (80%) und Validation (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # WICHTIG: Augmentation nur für das Training-Subset aktivieren
    # (Kleiner Hack, da random_split Subsets erstellt, greifen wir auf das zugrundeliegende Dataset zu)
    # Sauberer Weg: Wir setzen das Flag im DataLoader Loop oder wrappen es, 
    # aber hier zur Vereinfachung aktivieren wir Augmentation im Dataset und deaktivieren es bei Validierung nicht explizit im Code,
    # SONDERN: Wir erstellen zwei Datasets. Das ist sauberer.
    
    # Besserer Ansatz für sauberen Code:
    print(f"Lade Daten... ({len(full_dataset)} Samples)")
    train_data = PointCloudDataset(points_path, labels_path, augment=True) # Mit Augmentation
    val_data = PointCloudDataset(points_path, labels_path, augment=False)  # Ohne Augmentation
    
    # Indizes manuell splitten, damit wir keine Datenlecks haben
    indices = list(range(len(full_dataset)))
    # Wir nehmen an, die Daten sind shuffled vom Generator, aber sicher ist sicher:
    np.random.shuffle(indices)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(train_data, train_idx)
    val_dataset = torch.utils.data.Subset(val_data, val_idx)
    
    print(f"-> Training Samples: {len(train_dataset)}")
    print(f"-> Validation Samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)

    # 3. Modell Setup
    model = PointNetSegmentation(num_classes=2).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Gewichteter Loss: Wir geben dem Objekt (Klasse 1) etwas mehr Gewicht
    weights = torch.tensor(CONFIG["class_weights"]).to(CONFIG["device"])
    criterion = nn.NLLLoss(weight=weights)

    best_iou = 0.0 # Wir speichern basierend auf IoU, nicht Accuracy!

    for epoch in range(CONFIG["epochs"]):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            data, target = data.to(CONFIG["device"]), target.to(CONFIG["device"])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
        
        # --- VALIDATION PHASE (Alle paar Epochen oder jede Epoche) ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(CONFIG["device"]), target.to(CONFIG["device"])
                output = model(data)
                
                loss = criterion(output, target)
                val_loss += loss.item()
                
                acc, iou = calculate_metrics(output, target)
                val_acc += acc
                val_iou += iou

        # Durchschnittswerte
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(f"Epoche {epoch+1:02d} | "
              f"T-Loss: {avg_train_loss:.4f} | "
              f"V-Loss: {avg_val_loss:.4f} | "
              f"V-Acc: {avg_val_acc*100:.1f}% | "
              f"V-IoU (Obj): {avg_val_iou*100:.1f}%")

        # Checkpoint speichern (Nur wenn IoU besser wird!)
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"   >>> Neues bestes Modell gespeichert! (IoU: {best_iou*100:.1f}%)")

    print("--- Training beendet ---")
    print(f"Bestes Objekt-IoU auf Validation-Set: {best_iou*100:.2f}%")

if __name__ == "__main__":
    train()
