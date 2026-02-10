import open3d as o3d
import glob
import os
from alignment import align_scans
from inspection import inspect

# =================================================
# DATEIEN
# =================================================
CAD_FILE = r"CAD/kopfstueck.ply"
such_ordner = "scan_clean_pipeline"

# Scans suchen
SCAN_FILES = sorted(glob.glob(os.path.join(such_ordner, "*.ply")))
SCAN_NAMES = [os.path.splitext(os.path.basename(f))[0] for f in SCAN_FILES]

REPORT_CSV = "inspection_report.csv"
REWORK_CSV = "inspection_rework_points.csv"

# =================================================
# MAIN
# =================================================
def main():
    print("====================================")
    print("START: ALIGNMENT")
    print("====================================")

    # 1. Alignment durchführen
    cad_pcd, aligned_scans, metrics = align_scans(
        CAD_FILE, SCAN_FILES
    )

    # ---------------------------------------------------------
    # FILTER-LOGIK
    # ---------------------------------------------------------
    final_scans = []
    final_names = []
    
    print("\n------------------------------------")
    print("FILTERUNG NACH FITNESS (Min. 0.95)")
    print("------------------------------------")

    # Wir iterieren durch Scans, Metriken und Namen gleichzeitig
    for scan, metric, name in zip(aligned_scans, metrics, SCAN_NAMES):
        fitness_val = metric['fitness']
        
        if fitness_val >= 0.95:
            # Scan ist gut genug -> behalten
            final_scans.append(scan)
            final_names.append(name)
            print(f"[OK]  {name}: Fitness {fitness_val:.3f}")
        else:
            # Scan ist zu schlecht -> verwerfen
            print(f"[NOK] {name}: Fitness {fitness_val:.3f} -> WIRD ÜBERSPRUNGEN")

    # Sicherheitscheck: Wenn kein Scan übrig bleibt, abbrechen
    if not final_scans:
        print("\nACHTUNG: Kein Scan hat die Qualitätsanforderung (Fitness >= 0.95) erfüllt.")
        return

    print(f"\nEs verbleiben {len(final_scans)} Scans für die Prüfung.")

    print("\nViewer: Alignment (Nur gute Scans)")
    o3d.visualization.draw_geometries(
        [cad_pcd] + final_scans,
        window_name="Alignment – Gute Scans",
    )

    print("\n====================================")
    print("START: QS-PRÜFUNG")
    print("====================================")

    # 2. Inspection nur mit den gefilterten Scans
    colored_scans = inspect(
        cad_pcd=cad_pcd,
        scan_pcds=final_scans,     
        scan_names=final_names,    
        report_csv=REPORT_CSV,
        rework_csv=REWORK_CSV,
    )

    print("\nViewer: QS-Ergebnis")
    print(" Grün = OK")
    print(" Rot  = Grat XXX FAIL")
    print(" Blau = Anguss FAIL")
    print(" Gelb = OOO FAIL")

    o3d.visualization.draw_geometries(
        [cad_pcd] + colored_scans,
        window_name="QS Ergebnis – Zonenfarben",
    )

    print("\n====================================")
    print("FERTIG")
    print("====================================")


# =================================================
# ENTRY POINT
# =================================================
if __name__ == "__main__":
    main()



