import open3d as o3d
import numpy as np
import sys
import os
import copy
from inspection import GeometryInspector
from result_viewer import AlignmentViewer

# --- KONFIGURATION ---
SCAN_FILE = "front1_crt.ply"      
CAD_FILE = "kopfstueck.ply"      
CAD_SCALE = 0.001
VOXEL_SIZE = 0.002   

# 250k ist super für Speed & ausreichend für den Flip-Check
INSPECTION_POINT_COUNT = 250000 

# SCHWELLENWERT FÜR DEN FLIP-GUARD:
# Wie viele Punkte dürfen maximal in der Killzone sein, bevor wir Alarm schlagen?
# Da du 250.000 Punkte hast, sind 100-200 Punkte oft Rauschen. 
# Wenn aber 1000+ Punkte drin sind, liegt da echtes Material -> Falsch herum!
KILLZONE_POINT_LIMIT = 500 
# ---------------------

# (ZONES Liste unverändert)
ZONES = [
    {
        "name": "obere Seite (Killzone)",
        "center": [-0.02090, 0.00015, 0.00000],
        "extent": [0.00330, 0.07270, 0.01490],
        "R":      np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 
        "tolerance": 0.0005, 
    },
    {
        "name": "linke Zone",
        "center": [-0.03425, 0.03905, 0.00000],
        "extent": [0.02460, 0.00560, 0.02640],
        "R":      np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "rechte Zone",
        "center": [-0.04000, -0.03915, 0.00000],
        "extent": [0.03500, 0.00510, 0.03000],
        "R":      np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "Zone Mitte oben",
        "center": [-0.04165, 0.00000, 0.00000],
        "extent": [0.02350, 0.01080, 0.03550],
        "R":      np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "Zone rechts rotiert",
        "center": [-0.05430, 0.03315, 0.00000],
        "extent": [0.02000, 0.00500, 0.04500],
        "R":      np.array([[ 0.80385686, -0.59482279,  0.        ], [ 0.59482279,  0.80385686,  0.        ], [ 0.        ,  0.  ,  1.        ]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "Zone links rotiert",
        "center": [-0.06340, -0.03550, 0.00000],
        "extent": [0.01500, 0.01000, 0.03500],
        "R":      np.array([[ 0.46947156,  0.88294759,  0.        ], [-0.88294759,  0.46947156,  0.        ], [ 0.        ,  0.  ,  1.        ]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "Zone links unten rotiert",
        "center": [-0.06800, -0.02735, 0.00000],
        "extent": [0.00510, 0.00510, 0.01500],
        "R":      np.array([[ 0.9945219 ,  0.10452846,  0.        ], [-0.10452846,  0.9945219 ,  0.        ], [ 0.        ,  0.  ,  1.        ]]), 
        "tolerance": 0.0005,
    },
    {
        "name": "Zone rechts unten rotiert",
        "center": [-0.06800, 0.02700, 0.00000],
        "extent": [0.01000, 0.00500, 0.01500],
        "R":      np.array([[ 0.97029573, -0.2419219 ,  0.        ], [ 0.2419219 ,  0.97029573,  0.        ], [ 0.        ,  0.  ,  1.        ]]),
        "tolerance": 0.0005,
    }, 
    {
        "name": "Zone mitte rotiert1",
        "center": [-0.02475, 0.00920, 0.00000],
        "extent": [0.00510, 0.01500, 0.02700],
        "R":      np.array([[ 8.70349148e-01,  4.92434913e-01, -4.66016354e-04], [-4.92435133e-01,  8.70348794e-01, -7.85889605e-04], [ 1.85972925e-05,  9.13481174e-04,  9.99999583e-01]]),
        "tolerance": 0.0005,
    },
    {
        "name": "Zone mitte rotiert2",
        "center": [-0.02420, -0.01085, 0.00000],
        "extent": [0.00510, 0.01500, 0.02700],
        "R":      np.array([[ 8.70362243e-01, -4.92411797e-01,  4.33622706e-04], [ 4.92411987e-01,  8.70361872e-01, -8.04217050e-04], [ 1.85972925e-05,  9.13481174e-04,  9.99999583e-01]]),
        "tolerance": 0.0005,
    },
]

def load_geometry(path, scale=1.0):
    if not os.path.exists(path): return None, False
    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            mesh.scale(scale, center=(0,0,0))
            return mesh, True
    except: pass 
    pcd = o3d.io.read_point_cloud(path)
    pcd.scale(scale, center=(0,0,0))
    return pcd, False

def main():
    print("=== GRAT-INSPEKTION MIT FLIP-GUARD ===")
    
    # 1. Daten Laden
    print("\n[1/5] Lade Geometrie...")
    cad_raw, cad_is_mesh = load_geometry(CAD_FILE, scale=CAD_SCALE)
    scan_raw, scan_is_mesh = load_geometry(SCAN_FILE, scale=1.0) 
    
    if cad_raw is None or scan_raw is None:
        print("FEHLER: Dateien nicht gefunden.")
        return

    # Daten Vorbereitung
    if cad_is_mesh:
        cad_pcd_for_calc = cad_raw.sample_points_poisson_disk(20000)
    else:
        cad_pcd_for_calc = cad_raw

    if scan_is_mesh:
        scan_for_calc = scan_raw.sample_points_poisson_disk(50000)
    else:
        scan_for_calc = copy.deepcopy(scan_raw)

    # 2. INTELLIGENTE AUSRICHTUNG
    inspector = GeometryInspector(voxel_size=VOXEL_SIZE)
    
    result = inspector.register_blindly(
        scan_for_calc, 
        cad_pcd_for_calc, 
        attempts=50, # 50 Versuche für maximale Sicherheit
        min_fitness_threshold=0.70 
    )
    
    if result is None:
        print("FEHLER: Suche komplett fehlgeschlagen.")
        return

    print(f" -> GEWINNER FITNESS: {result.fitness:.4f}")

    # 3. TRANSFORMATION ANWENDEN
    final_transform = result.transformation
    scan_raw.transform(final_transform)
    
    # Inspektion Wolke erstellen
    print(f"\n[3/5] Erstelle Inspektions-Wolke ({INSPECTION_POINT_COUNT} Punkte)...")
    if scan_is_mesh:
        final_vis_pcd = scan_raw.sample_points_poisson_disk(INSPECTION_POINT_COUNT)
    else:
        final_vis_pcd = copy.deepcopy(scan_raw)

    final_vis_pcd.paint_uniform_color([0, 0.8, 0]) 
    colors_np = np.asarray(final_vis_pcd.colors)

    # --- 4. NEU: PLAUSIBILITÄTS-CHECK (FLIP GUARD) ---
    print("\n[4/5] Checke Plausibilität (Flip-Guard)...")
    
    flip_detected = False
    
    # Wir suchen speziell nach deiner Killzone
    for zone in ZONES:
        if "killzone" in zone["name"].lower():
            # Box erstellen
            box = o3d.geometry.OrientedBoundingBox(
                center=np.array(zone["center"]), R=zone["R"], extent=np.array(zone["extent"])
            )
            # Zähle Punkte in der Box
            indices = box.get_point_indices_within_bounding_box(final_vis_pcd.points)
            point_count = len(indices)
            
            print(f" -> Prüfe Zone '{zone['name']}': {point_count} Punkte gefunden.")
            
            if point_count > KILLZONE_POINT_LIMIT:
                flip_detected = True
                print("\n" + "!"*60)
                print(f"WARNUNG: {point_count} Punkte in der Killzone gefunden!")
                print(f"Limit ist {KILLZONE_POINT_LIMIT}.")
                print("DAS BAUTEIL LIEGT WAHRSCHEINLICH FALSCH HERUM (GESPIEGELT)!")
                print("Trotz hoher Fitness (Symmetrie-Falle).")
                print("!"*60 + "\n")
            else:
                print(" -> Plausibilität OK. Killzone ist sauber.")

    # 5. ZONEN PRÜFUNG (Messung)
    print(" -> Prüfe Toleranzen...")
    scene = None
    if cad_is_mesh:
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(cad_raw)
        scene.add_triangles(mesh_t)
    
    for zone in ZONES:
        try:
            box = o3d.geometry.OrientedBoundingBox(
                center=np.array(zone["center"]), R=zone["R"], extent=np.array(zone["extent"])
            )
            indices = box.get_point_indices_within_bounding_box(final_vis_pcd.points)
            if len(indices) == 0: continue
            
            if scene:
                zone_points = np.asarray(final_vis_pcd.points)[indices]
                query = o3d.core.Tensor.from_numpy(zone_points.astype(np.float32))
                signed_dist = scene.compute_signed_distance(query).numpy()
                
                for k, idx in enumerate(indices):
                    if abs(signed_dist[k]) > zone["tolerance"]:
                        colors_np[idx] = [1, 0, 0]
            else:
                for idx in indices: colors_np[idx] = [1, 0, 0]
        except: continue

    final_vis_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    
    # 6. VIEWER
    window_title = f"Inspektion (Fit: {result.fitness:.2f})"
    if flip_detected:
        window_title += " - ACHTUNG: VERDACHT AUF SPIEGELUNG!"
        
    print(f"\n[5/5] Starte Viewer: {window_title}")
    viewer = AlignmentViewer(final_vis_pcd, cad_raw, window_name=window_title)
    viewer.run()

if __name__ == "__main__":
    main()
