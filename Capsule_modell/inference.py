import torch
import numpy as np
import open3d as o3d
import os
from segmentation_model import PointNetSegmentation

# KONFIGURATION
MODEL_PATH = "best_segmentation_model.pth"
DATA_PATH = "training_data/train_points.npy" # Wir klauen uns ein Sample aus den Trainingsdaten zum Testen
SAMPLE_IDX = 0 # Welches Beispiel wollen wir ansehen? (0 bis 1999)

def visualize():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Lade Modell auf {device}...")

    # Modell initialisieren und Gewichte laden
    model = PointNetSegmentation(num_classes=2).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Fehler: Modell '{MODEL_PATH}' nicht gefunden. Erst trainieren!")
        return

    # WICHTIG: map_location sorgt dafür, dass ein auf GPU trainiertes Modell auch auf CPU läuft
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # WICHTIG: Schaltet Dropout und BatchNorm in den Test-Modus

    # 2. Daten laden
    if not os.path.exists(DATA_PATH):
        print("Keine Daten gefunden. Generiere erst welche.")
        return
        
    all_points = np.load(DATA_PATH)
    sample_points = all_points[SAMPLE_IDX] # Shape: (2048, 3)
    
    # Vorbereiten für PyTorch: (1, 3, 2048)
    input_tensor = torch.from_numpy(sample_points).float().to(device)
    input_tensor = input_tensor.transpose(1, 0).unsqueeze(0) # [N,3] -> [3,N] -> [1,3,N]

    # 3. Vorhersage (Inference)
    print("Starte Segmentation...")
    with torch.no_grad():
        output = model(input_tensor) # Log Softmax Output
        pred_choice = output.max(1)[1] # Argmax -> Klassen Index (0 oder 1)
        
    pred_np = pred_choice.cpu().numpy()[0] # Zurück zu Numpy Array (2048,)

    # 4. Visualisierung
    print(f"Gefundene Bauteil-Punkte: {np.sum(pred_np == 1)} von {len(pred_np)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_points)

    # FARBEN SETZEN
    # Wir erstellen eine Farbmatrix: Grau für Hintergrund, Rot für Bauteil
    colors = np.zeros((len(sample_points), 3))
    
    # Hintergrund (Label 0) -> Dunkelgrau
    colors[pred_np == 0] = [0.3, 0.3, 0.3] 
    
    # Bauteil (Label 1) -> Rot
    colors[pred_np == 1] = [1.0, 0.0, 0.0] 

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("Öffne 3D Fenster...")
    o3d.visualization.draw_geometries([pcd], window_name="KI Vorhersage")

if __name__ == "__main__":
    visualize()
