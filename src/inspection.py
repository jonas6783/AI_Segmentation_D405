import numpy as np
import open3d as o3d
import csv
import json

from zones import (
    picked_B,
    picked_A_center,
    grat_xxx_zones,
    ooo_zones,
    p1, p2
)

# =================================================
# TOLERANZEN 
# =================================================
import json
with open("tolerances.json") as f:
    TOL = json.load(f)


SLOT_MIN = 12.4
SLOT_MAX = 13.0

# =================================================
# PADDING 
# =================================================
PAD_OOO = np.array([1.0, 1.0, 1.0])
PAD_GRAT = np.array([1.5, 1.5, 1.5])

# =================================================
# FARBEN 
# =================================================
COLOR_OK = [0.0, 1.0, 0.0]  # Grün

ZONE_COLORS_FAIL = {
    "GRAT_XXX": [1.0, 0.0, 0.0],  # Rot
    "ANGUSS_A": [0.0, 0.0, 1.0],  # Blau
    "ANGUSS_B": [0.0, 0.0, 1.0],  # Blau
    "OOO":      [1.0, 1.0, 0.0],  # Gelb
}

# =================================================
# HILFSFUNKTIONEN
# =================================================
def in_box(p, bmin, bmax):
    return np.all(p >= bmin) and np.all(p <= bmax)

def build_box(pts, pad):
    return pts.min(axis=0) - pad, pts.max(axis=0) + pad

# =================================================
# INSPECTION + ROBOTER-REWORK
# =================================================
def inspect(
    cad_pcd,
    scan_pcds,
    scan_names,
    report_csv="inspection_report.csv",
    rework_csv="inspection_rework_points.csv"
):
    cad_pts = np.asarray(cad_pcd.points)
    cad_tree = o3d.geometry.KDTreeFlann(cad_pcd)

    def cad_dist(p):
        _, idx, _ = cad_tree.search_knn_vector_3d(p, 1)
        return np.linalg.norm(p - cad_pts[idx[0]])

    # =================================================
    # ZONEN DEFINIEREN
    # =================================================
    # Anguss B
    zoneB_min = picked_B.min(axis=0)
    zoneB_max = picked_B.max(axis=0)

    # Anguss A
    half_size = (zoneB_max - zoneB_min) / 2.0
    zoneA_min = picked_A_center - half_size
    zoneA_max = picked_A_center + half_size

    # OOO
    ooo_boxes = [build_box(z, PAD_OOO) for z in ooo_zones]

    # =================================================
    # CSV-VORBEREITUNG
    # =================================================
    report_rows = [
        ["scan", "zone", "max_deviation_mm", "tolerance_mm", "status"]
    ]

    rework_rows = [
        ["scan", "zone", "x_mm", "y_mm", "z_mm", "deviation_mm"]
    ]

    # =================================================
    # PRÜFUNG
    # =================================================
    for scan, scan_name in zip(scan_pcds, scan_names):
        scan_pts = np.asarray(scan.points)

        max_dev = {
            "OOO": 0.0,
            "ANGUSS_A": 0.0,
            "ANGUSS_B": 0.0,
            "GRAT_XXX": 0.0,
        }

        colors = []

        for p in scan_pts:
            d = cad_dist(p)

            # ---------- PRIORITÄT ----------
            zone = "GRAT_XXX"
            tol = TOL["GRAT_XXX"]

            for zmin, zmax in ooo_boxes:
                if in_box(p, zmin, zmax):
                    zone = "OOO"
                    tol = TOL["OOO"]
                    break

            if in_box(p, zoneA_min, zoneA_max):
                zone = "ANGUSS_A"
                tol = TOL["ANGUSS_A"]

            elif in_box(p, zoneB_min, zoneB_max):
                zone = "ANGUSS_B"
                tol = TOL["ANGUSS_B"]

            # ---------- AUSWERTUNG ----------
            max_dev[zone] = max(max_dev[zone], d)

            if d > tol:
                
                colors.append(ZONE_COLORS_FAIL[zone])

                rework_rows.append([
                    scan_name,
                    zone,
                    f"{p[0]:.3f}",
                    f"{p[1]:.3f}",
                    f"{p[2]:.3f}",
                    f"{d:.3f}",
                ])
            else:
                colors.append(COLOR_OK)

        scan.colors = o3d.utility.Vector3dVector(colors)

        # ---------- REPORT CSV ----------
        for z, v in max_dev.items():
            status = "OK" if v <= TOL[z] else "NOK"
            report_rows.append([
                scan_name,
                z,
                f"{v:.3f}",
                TOL[z],
                status
            ])

        # ---------- SCHLITZ E ----------
        slot_width = np.linalg.norm(p2 - p1)
        slot_status = "OK" if SLOT_MIN <= slot_width <= SLOT_MAX else "NOK"

        report_rows.append([
            scan_name,
            "SCHLITZ_E",
            f"{slot_width:.3f}",
            "12.4–13.0",
            slot_status
        ])

    # =================================================
    # CSV SCHREIBEN
    # =================================================
    with open(report_csv, "w", newline="") as f:
        csv.writer(f).writerows(report_rows)

    with open(rework_csv, "w", newline="") as f:
        csv.writer(f).writerows(rework_rows)

    print("\n========================================")
    print("QS + ROBOTER-REWORK ABGESCHLOSSEN")
    print(f"QS-Report:    {report_csv}")
    print(f"Rework-Punkte:{rework_csv}")
    print("----------------------------------------")
    print("Rework-CSV enthält NUR Punkte,")
    print("die der Roboter wegschleifen muss.")
    print("Koordinaten sind in mm (CAD-Koordinatensystem).")
    print("========================================\n")

    return scan_pcds

