import open3d as o3d
import numpy as np
import os
from pathlib import Path

# ==========================================
# KONFIGURATION
# ==========================================
CAD_FILE = "cad/kopfstueck.ply"  
OUTPUT_DIR = "training_data"
NUM_SAMPLES = 2000               
POINTS_PER_SAMPLE = 2048         

# Verteilung der Punkte
NOISE_RATIO = 0.35               # 35% Hintergrund (Tisch + Rauschen)
POINTS_OBJ = int(POINTS_PER_SAMPLE * (1 - NOISE_RATIO))
POINTS_BG = POINTS_PER_SAMPLE - POINTS_OBJ

# Kamerasimulation Parameter
# Radius-Faktor: Wie "groß" ist der Sichtkegel. Höher = mehr Punkte sichtbar.
HPR_RADIUS_FACTOR = 100.0        

def get_random_rotation_matrix():
    """Erzeugt eine zufällige Rotationsmatrix."""
    angles = np.random.rand(3) * 2 * np.pi
    R = o3d.geometry.get_rotation_matrix_from_xyz(angles)
    return R

def normalize_pc(points):
    """
    Verschiebt PC in den Ursprung und skaliert in den Einheitswürfel [-1, 1].
    Behält die Aspekte bei (kein Verzerren).
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    # Den weitesten Punkt finden für Skalierung
    furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2, axis=-1)))
    if furthest_distance > 0:
        points /= furthest_distance
    return points

def sample_visible_surface(mesh, num_points_needed):
    """
    Simuliert eine 3D-Kamera:
    1. Rotiert das Mesh zufällig.
    2. Entfernt verdeckte Punkte (Rückseite), die die Kamera nicht sieht.
    3. Resampelt auf exakt 'num_points_needed'.
    """
    # A) Mesh kopieren und rotieren
    mesh_copy = o3d.geometry.TriangleMesh(mesh)
    R = get_random_rotation_matrix()
    mesh_copy.rotate(R, center=(0, 0, 0))
    
    # B) Erstmal dicht sampeln, um eine gute Oberfläche zu haben
    # Wir sampeln 4x so viel, da ca. 50% durch Sichtbarkeit wegfallen werden
    pcd = mesh_copy.sample_points_poisson_disk(num_points_needed * 4)
    
    # C) Hidden Point Removal (Kamera-Simulation)
    # Wir nehmen an, die Kamera schaut von +Z auf das Objekt
    min_bound = mesh_copy.get_min_bound()
    max_bound = mesh_copy.get_max_bound()
    center = mesh_copy.get_center()
    diameter = np.linalg.norm(max_bound - min_bound)
    
    # Kamera Positionieren: Weit genug weg auf der Z-Achse + leichter Jitter
    camera_pos = [0, 0, diameter * 3] 
    radius = diameter * HPR_RADIUS_FACTOR
    
    # Open3D berechnet sichtbare Punkte
    _, pt_map = pcd.hidden_point_removal(camera_pos, radius)
    pcd_visible = pcd.select_by_index(pt_map)
    points_visible = np.asarray(pcd_visible.points)
    
    # D) Auf exakte Anzahl bringen (Resampling)
    current_count = len(points_visible)
    
    if current_count == 0:
        # Fallback: Sollte nicht passieren, aber sicher ist sicher
        return np.zeros((num_points_needed, 3))
        
    if current_count >= num_points_needed:
        # Wir haben zu viele -> Zufällig auswählen (Downsampling)
        choice = np.random.choice(current_count, num_points_needed, replace=False)
        final_points = points_visible[choice]
    else:
        # Wir haben zu wenige -> Punkte duplizieren (Upsampling mit Replacement)
        # Das ist okay, PointNet verkraftet doppelte Punkte
        choice = np.random.choice(current_count, num_points_needed, replace=True)
        final_points = points_visible[choice]
        
    # E) Sensor-Rauschen hinzufügen (Jitter)
    # 10cm Objekt -> Jitter von 0.5mm - 1mm ist realistisch
    jitter = np.random.normal(0, 0.002 * diameter, final_points.shape)
    final_points += jitter
    
    # Um das Training zu erschweren, verschieben wir das Objekt auch leicht im Raum
    # (Translation), damit es nicht immer perfekt zentriert ist bevor es normalisiert wird
    shift = np.random.uniform(-diameter*0.1, diameter*0.1, 3)
    final_points += shift
    
    return final_points, mesh_copy # Mesh zurückgeben für Bounding Box Berechnung

def generate_table_background(mesh_bounds, num_points):
    """
    Erzeugt eine Ebene (Tisch) unter dem Objekt sowie etwas Zufallsrauschen.
    """
    min_b, max_b = mesh_bounds
    width = (max_b[0] - min_b[0]) * 2.0 # Tisch ist größer als Objekt
    depth = (max_b[1] - min_b[1]) * 2.0
    
    # Z-Höhe: Knapp unter dem Objekt (min_z)
    z_level = min_b[2] - (max_b[2] - min_b[2]) * 0.05 
    
    # 70% der Background-Punkte sind Tisch, 30% sind "Luft-Rauschen"
    num_plane = int(num_points * 0.7)
    num_noise = num_points - num_plane
    
    # 1. Ebene erzeugen (X, Y zufällig, Z fix)
    plane_x = np.random.uniform(min_b[0]-width/2, max_b[0]+width/2, num_plane)
    plane_y = np.random.uniform(min_b[1]-depth/2, max_b[1]+depth/2, num_plane)
    plane_z = np.full(num_plane, z_level)
    
    # Leichtes Rauschen auf dem Tisch (Unebenheit)
    plane_z += np.random.normal(0, 0.005, num_plane)
    
    points_plane = np.column_stack((plane_x, plane_y, plane_z))
    
    # 2. Random Scatter ("Sensorfehler" in der Luft)
    noise_min = min_b * 1.5
    noise_max = max_b * 1.5
    points_noise = np.random.uniform(noise_min, noise_max, (num_noise, 3))
    
    return np.vstack((points_plane, points_noise))

def generate_dataset():
    if not os.path.exists(CAD_FILE):
        print(f"Fehler: CAD Datei nicht gefunden unter {CAD_FILE}")
        return
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Lade CAD: {CAD_FILE}...")
    mesh_original = o3d.io.read_triangle_mesh(CAD_FILE)
    # Mesh in den Ursprung schieben für einfachere Berechnung
    mesh_original.translate(-mesh_original.get_center())
    
    print(f"Starte Generierung von {NUM_SAMPLES} Samples (Single-View Simulation)...")
    print(f" -> Objekt: ca. {POINTS_OBJ} Punkte (sichtbare Fläche)")
    print(f" -> Background: {POINTS_BG} Punkte (Tisch + Noise)")

    all_points = []
    all_labels = []

    for i in range(NUM_SAMPLES):
        # 1. Sichtbare Objektpunkte generieren
        # Wir erhalten auch das rotierte Mesh zurück, um die BoundingBox für den Hintergrund zu kennen
        points_obj, rotated_mesh = sample_visible_surface(mesh_original, POINTS_OBJ)
        labels_obj = np.ones(len(points_obj)) # Label 1
        
        # 2. Hintergrund generieren basierend auf Position des Objekts
        bounds = (rotated_mesh.get_min_bound(), rotated_mesh.get_max_bound())
        points_bg = generate_table_background(bounds, POINTS_BG)
        labels_bg = np.zeros(len(points_bg)) # Label 0
        
        # 3. Zusammenfügen
        points_combined = np.vstack((points_obj, points_bg))
        labels_combined = np.concatenate((labels_obj, labels_bg))
        
        # 4. Shuffling (Wichtig!)
        indices = np.arange(len(points_combined))
        np.random.shuffle(indices)
        points_combined = points_combined[indices]
        labels_combined = labels_combined[indices]
        
        # 5. Normalisierung (Essentiell für PointNet)
        points_combined = normalize_pc(points_combined)
        
        all_points.append(points_combined)
        all_labels.append(labels_combined)
        
        if (i+1) % 100 == 0:
            print(f"Erzeugt: {i+1}/{NUM_SAMPLES}")

    # Speichern
    print("Speichere Daten...")
    np.save(f"{OUTPUT_DIR}/train_points.npy", np.array(all_points, dtype=np.float32))
    np.save(f"{OUTPUT_DIR}/train_labels.npy", np.array(all_labels, dtype=np.int64)) # Labels müssen int/long sein
    
    print("Fertig! Daten gespeichert in:", OUTPUT_DIR)
    print("Output Shape:", np.array(all_points).shape)

if __name__ == "__main__":
    generate_dataset()
