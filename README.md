# AI_Segmentation_D405

Dieses System automatisiert die Qualitätsprüfung von Bauteilen mittels 3D-Scanning. Es kombiniert Hardware-Steuerung, KI-gestützte Bildverarbeitung und präzise 3D-Metrologie in einer durchgängigen Pipeline.

##  Projektstruktur

| Kategorie | Dateien | Beschreibung |
| :--- | :--- | :--- |
| **Workflow** | `main.py`, `main2.py`, `config.py` | Steuerung der Analyse und des Scan-Prozesses. |
| **Algorithmen** | `alignment.py`, `inspection.py`, `processor.py` | Registrierung, Zonen-Abweichungsprüfung und KI-Segmentierung. |
| **Hardware** | `camera.py`, `turntable.py`, `calibration.py` | Schnittstellen für RealSense D405, G-Code Drehteller und ArUco-Tracking. |
| **Daten** | `Cad/`, `tolerances.json` | Referenzmodelle und konfigurierbare QS-Grenzwerte. |

##  Kernfunktionen

### 1. Datenakquise & KI-Processing
In der Akquise-Phase (`main2.py`) wird jedes aufgenommene Frame durch eine KI-gestützte Segmentierung (`rembg`) von Hintergrundrauschen befreit. Die Punktwolken werden basierend auf der ArUco-Kalibrierung (`calibration.py`) direkt im Weltkoordinatensystem des Drehtellers platziert.

### 2. GPU-beschleunigtes Alignment
Das System nutzt eine zweistufige Registrierung in `alignment.py`:
* **Global**: RANSAC-basiertes Feature-Matching für die Grobausrichtung.
* **Lokal**: Point-to-Plane ICP (Tensor-basiert) für maximale Präzision auf der GPU.

### 3. Zonen-basierte Inspektion
Die `inspection.py` vergleicht den fertigen Scan mit dem CAD-Modell. Dabei werden spezifische Regionen (z. B. "GRAT_XXX" oder "ANGUSS") individuell gegen die in `tolerances.json` definierten Werte geprüft.

##  Code-Beispiel: Inspektion starten

```python
"""
Beispiel für die Durchführung einer Inspektion nach erfolgreichem Alignment.
"""
from inspection import inspect
import open3d as o3d

# CAD-Referenz und Scans laden
cad_model = o3d.io.read_point_cloud("Cad/kopfstueck.ply")
scans = [o3d.io.read_point_cloud("scan_clean_pipeline/scan_000.ply")]

# Inspektion ausführen
colored_results = inspect(
    cad_pcd=cad_model,
    scan_pcds=scans,
    scan_names=["Bauteil_01"]
)
