#  Automatisierter 360°-Prüfstand zur optischen Qualitätskontrolle

> **Entwicklung eines automatisierten 360°-Prüfstands zur optischen Qualitätskontrolle mittels Intel RealSense D405 und CAD-basiertem Soll-Ist-Abgleich**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.17+-green.svg)](http://www.open3d.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hochschule Niederrhein](https://img.shields.io/badge/Hochschule-Niederrhein-red.svg)](https://www.hs-niederrhein.de/)

**Kurs:** KI-Anwendungen im betrieblichen Umfeld | Wintersemester 2025/26  
**Autoren:** Lukas Kennerknecht · Jonas Damek  
**Hochschule:** Hochschule Niederrhein – University of Applied Sciences

---

## Inhaltsverzeichnis

- [Projektbeschreibung](#-projektbeschreibung)
- [Systemvoraussetzungen](#-systemvoraussetzungen)
- [Hardware-Setup](#-hardware-setup)
- [Installation](#-installation)
- [Konfiguration](#️-konfiguration)
- [Verwendung](#-verwendung)
- [Projektstruktur](#-projektstruktur)
- [Funktionsweise](#-funktionsweise)
- [Qualitätszonen & Toleranzen](#-qualitätszonen--toleranzen)
- [Ausgabe & Reports](#-ausgabe--reports)
- [Bekannte Limitationen](#-bekannte-limitationen)
- [Lizenz](#-lizenz)

---

##  Projektbeschreibung

Dieses Projekt realisiert einen vollautomatisierten Prototyp für eine **3D-Prüfstation** zur optischen Qualitätskontrolle von Gussbauteilen. Das System erfasst ein Bauteil mittels einer **Intel RealSense D405** Tiefenkamera und einem computergesteuerten Drehteller vollständig in 360° und vergleicht die gewonnene Punktwolke automatisiert mit einem **CAD-Soll-Modell**.

### Was das System kann:

- ✅ **Vollständige 360°-Erfassung** des Bauteils in definierten 20°-Schritten
- ✅ **KI-basierte Hintergrundentfernung** (rembg / ISNET-Modell) für saubere Punktwolken
- ✅ **Automatisches CAD-Alignment** via RANSAC + Point-to-Plane ICP (GPU-beschleunigt)
- ✅ **Symmetrie-Erkennung** durch implementierte Flip-Logik
- ✅ **Zonenbasierte Toleranzprüfung** mit individuellen Grenzwerten pro Funktionsbereich
- ✅ **Automatische Report-Generierung** als maschinenlesbare CSV-Dateien
- ✅ **3D-Heatmap-Visualisierung** mit farbkodierter Fehlerdarstellung
- ✅ **Rework-Koordinaten** für automatisierte Roboter-Nachbearbeitung

---

##  Systemvoraussetzungen

### Software
| Komponente | Mindestversion |
|---|---|
| Python | 3.8+ |
| CUDA Toolkit | 11.x+ (optional, für GPU-Beschleunigung) |
| Betriebssystem | Ubuntu 20.04+ / Windows 10+ |

### Hardware
| Komponente | Spezifikation |
|---|---|
| Intel RealSense D405 | Stereo-Tiefenkamera |
| Motorisierter Drehteller | G-Code-fähig, Seriell |
| Workstation | mind. 8 GB RAM, NVIDIA GPU empfohlen |

---

##  Hardware-Setup

```
         ┌─────────────────────────────────────┐
         │                                     │
         │     Intel RealSense D405            │
         │     (fest positioniert, ~15 cm)     │
         │                                     │
         └──────────────────┬──────────────────┘
                            │ USB 3.0
                            ▼
         ┌─────────────────────────────────────┐
         │          Workstation                │
         │     (Python + CUDA + Open3D)        │
         └──────────┬──────────────────────────┘
                    │ USB Serial (/dev/ttyUSB0)
                    ▼
         ┌─────────────────────────────────────┐
         │       Motorisierter Drehteller      │
         │     ┌─────────────────────┐         │
         │     │   [ArUco Marker]    │         │
         │     │    [  Bauteil  ]    │         │
         │     └─────────────────────┘         │
         └─────────────────────────────────────┘
```

### Aufbau-Schritte:
1. **Kamera** fest und stabil über dem Drehteller positionieren (optimaler Abstand: 7–20 cm)
2. **ArUco-Marker** (4×4_50 Dictionary, 25 mm) mittig auf den Drehteller legen
3. **Bauteil** auf dem Marker-Zentrum positionieren
4. **Intel RealSense D405** via USB 3.0 an die Workstation anschließen
5. **Drehteller** via USB-Seriell anschließen

---

##  Installation

### 1. Repository klonen
```bash
git clone https://github.com/lukas32123/AI_Segmentation_D405.git
cd AI_Segmentation_D405
```

### 2. Virtuelle Umgebung erstellen (empfohlen)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 4. CUDA-Version von Open3D installieren (optional, für GPU-Beschleunigung)
```bash
# Für CUDA 11.x
pip install open3d-cuda

# Ohne CUDA (CPU-Fallback ist automatisch aktiv)
pip install open3d
```

### 5. Installation prüfen
```bash
python -c "import open3d as o3d; print('Open3D:', o3d.__version__)"
python -c "import pyrealsense2 as rs; print('RealSense SDK OK')"
python -c "from rembg import remove; print('rembg OK')"
```

---

##  Konfiguration

Alle Systemparameter werden **zentral** in `config.py` und `tolerances.json` verwaltet. Der Quellcode muss für Anpassungen **nicht** geändert werden.

### `config.py` – Systemparameter

```python
# Kamera
CAMERA_WIDTH  = 1280        # Auflösung Breite (Pixel)
CAMERA_HEIGHT = 720         # Auflösung Höhe (Pixel)
CAMERA_FPS    = 30          # Framerate

# Hardware (Serial)
SERIAL_PORT   = '/dev/ttyUSB0'   # Linux: /dev/ttyUSB0 | Windows: 'COM3'
BAUDRATE      = 115200

# ArUco Marker
MARKER_SIZE   = 0.025       # Marker-Kantenlänge in Meter (25 mm)

# Scan-Parameter
ROTATION_STEP = 20.0        # Drehwinkel pro Schritt in Grad
TOTAL_ANGLE   = 360.0       # Gesamtwinkel für einen vollständigen Scan

# ROI-Box (Bereich um den Marker-Mittelpunkt)
BOX_HALF_X    = 0.1         # ±10 cm in X
BOX_HALF_Y    = 0.1         # ±10 cm in Y
BOX_HALF_Z    = 0.1         # ±10 cm in Z

# Filter-Optionen
USE_AI_MASK        = True   # KI-Hintergrundentfernung (empfohlen: True)
USE_DBSCAN         = False  # Cluster-basierte Filterung
USE_OUTLIER_REMOVAL= False  # Statistische Ausreißer-Entfernung

# Output
OUTPUT_FOLDER = "scan_clean_pipeline"
```

### `tolerances.json` – Zonenspezifische Toleranzen

```json
{
  "OOO":      0.0,    // Hochpräzise Funktionsflächen (keinerlei Abweichung)
  "GRAT_XXX": 1.2,    // Gratbereiche (bis 1.2 mm toleriert)
  "ANGUSS_A": 1.4,    // Angussbereich A (bis 1.4 mm toleriert)
  "ANGUSS_B": 1.0     // Angussbereich B (bis 1.0 mm toleriert)
}
```

> **Hinweis:** Toleranzwerte in `tolerances.json` können jederzeit angepasst werden, ohne den Quellcode zu ändern.

---

##  Verwendung

Das System ist in **zwei unabhängige Phasen** aufgeteilt:

---

### Phase 1: 3D-Scan durchführen

```bash
cd src
python main_scanning.py
```

**Ablauf:**
```
--- D405 Scanner: Marker-Mode ---
Marker-Kalibrierung starten? (j/n): j

[Calibration] Suche Marker (2.5 cm) als Nullpunkt...
[Calibration] Nullpunkt gefunden bei: [0.003, 0.001, 0.152]

Stelle jetzt das Objekt auf den Marker.
Drücke ENTER zum Starten...

[Camera] High Density Modus aktiviert.
[Processor] KI Modell laden (ISNET)...

Scan 0°...
Scan 20°...
Scan 40°...
...
Scan 340°...

Scan Vorgang Abgeschlossen
```

**Ergebnis:** `scan_clean_pipeline/scan_000.ply` bis `scan_340.ply` (18 Dateien)

---

### Phase 2: Qualitätsprüfung (Alignment + Inspektion)

```bash
cd src
python main_processing.py
```

**Ablauf:**
```
====================================
START: ALIGNMENT
====================================
Aligning scan_clean_pipeline/scan_000.ply ...
Aligning scan_clean_pipeline/scan_020.ply ...
...

------------------------------------
FILTERUNG NACH FITNESS (Min. 0.95)
------------------------------------
[OK]  scan_000: Fitness 0.971
[OK]  scan_020: Fitness 0.968
[NOK] scan_040: Fitness 0.923 -> WIRD ÜBERSPRUNGEN
...

Es verbleiben X Scans für die Prüfung.

====================================
START: QS-PRÜFUNG
====================================

========================================
QS + ROBOTER-REWORK ABGESCHLOSSEN
QS-Report:    inspection_report.csv
Rework-Punkte:inspection_rework_points.csv
========================================

FERTIG
```

**Visualisierung:**
- Ein Open3D-Fenster öffnet sich automatisch
- 🟢 **Grün** = Innerhalb der Toleranz (OK)
- 🔴 **Rot** = Toleranzüberschreitung im Gratbereich
- 🔵 **Blau** = Toleranzüberschreitung an Angussstellen
- 🟡 **Gelb** = Fehler an hochsensiblen Funktionsflächen

---

##  Projektstruktur

```
AI_Segmentation_D405/
│
├── src/                              # Gesamte Programm-Logik
│   ├── main_scanning.py              #  Phase 1: Scan-Orchestrator
│   ├── main_processing.py            #  Phase 2: QS-Pipeline
│   ├── camera.py                     # Intel RealSense D405 Klasse
│   ├── turntable.py                  # Drehteller-Steuerung (G-Code)
│   ├── calibration.py                # ArUco-Marker Kalibrierung
│   ├── processor.py                  # Punktwolken-Verarbeitungs-Pipeline
│   ├── alignment.py                  # RANSAC + ICP Alignment
│   └── inspection.py                 # Zonenbasierte QS-Inspektion
│
├── CAD/
│   └── kopfstueck.ply                # Referenz-CAD-Modell (Soll-Geometrie)
│
├── scan_clean_pipeline/              #  Generiert nach Phase 1
│   ├── scan_000.ply                  # Scan bei 0°
│   ├── scan_020.ply                  # Scan bei 20°
│   └── ...                           # bis scan_340.ply
│
├── inspection_report.csv             #  Generiert nach Phase 2
├── inspection_rework_points.csv      #  Rework-Koordinaten für Roboter
│
├── config.py                         #  Globale Systemparameter
├── tolerances.json                   #  Zonenbasierte Toleranzwerte
├── requirements.txt                  # Python-Abhängigkeiten
├── .gitignore                        # Git-Ausschlüsse
└── README.md                         # Diese Datei
```

---

##  Funktionsweise

### Gesamtablauf

```
┌─────────────────────────────────────────────────────────────────┐
│                        PHASE 1: SCANNING                        │
│                                                                  │
│  ArUco Marker           Kamera             Drehteller            │
│  Kalibrierung    →    start()        +    rotate(20°)           │
│  (Nullpunkt)          get_frame_data()     (18 Schritte)         │
│       │                    │                                     │
│       └───────────► PointCloudProcessor                          │
│                           │                                      │
│                    1. KI-Maskierung (rembg)                      │
│                    2. RGBD → Punktwolke                          │
│                    3. Marker-Zentrierung                         │
│                    4. Rotation anwenden                          │
│                    5. ROI Box-Filter                             │
│                    6. Voxel-Downsampling (0.5mm)                 │
│                    7. Speichern → scan_XXX.ply                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       PHASE 2: PROCESSING                        │
│                                                                  │
│  scan_*.ply + CAD/kopfstueck.ply                                 │
│         │                                                        │
│         ▼                                                        │
│    align_scans()                                                 │
│    ├── FPFH Feature Matching                                     │
│    ├── RANSAC Grob-Ausrichtung                                   │
│    ├── Flip-Logik (Symmetrie-Korrektur)                          │
│    └── Point-to-Plane ICP (GPU)                                  │
│         │                                                        │
│         ▼                                                        │
│    Quality Gate: Fitness ≥ 0.95 ?                                │
│    ├── JA  → weiter zur Inspektion                               │
│    └── NEIN → Scan verwerfen                                     │
│         │                                                        │
│         ▼                                                        │
│    inspect()                                                     │
│    ├── KDTree Distanzberechnung (Ist → CAD-Soll)                 │
│    ├── Zonen-Zuordnung & Toleranz-Prüfung                        │
│    ├── Farbkodierung (Grün/Rot/Blau/Gelb)                        │
│    └── CSV-Report-Generierung                                    │
│         │                                                        │
│         ▼                                                        │
│       inspection_report.csv                                      │
│       inspection_rework_points.csv                               │
│        Open3D 3D-Visualisierung                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

##  Qualitätszonen & Toleranzen

Das System prüft das Bauteil zonenspezifisch mit individuellen Toleranzwerten:

| Zone | Toleranz | Farbe (Fail) | Beschreibung |
|------|----------|--------------|--------------|
| `OOO` | **0.0 mm** | 🟡 Gelb | Hochpräzise Funktionsflächen – keinerlei Abweichung zulässig |
| `GRAT_XXX` | **1.2 mm** | 🔴 Rot | Gratbereiche an Trennebenen der Gussform |
| `ANGUSS_A` | **1.4 mm** | 🔵 Blau | Angussbereich A – fertigungsbedingte Materialüberstände |
| `ANGUSS_B` | **1.0 mm** | 🔵 Blau | Angussbereich B – engere Toleranz als A |
| `SCHLITZ_E` | **12.4–13.0 mm** | – | Geometrische Maßprüfung (euklidische Distanz) |

> **Quality Gate:** Scans mit einem Fitness-Score < 0.95 werden automatisch verworfen und nicht für die Toleranzprüfung herangezogen.

---

##  Ausgabe & Reports

### `inspection_report.csv` – QS-Prüfbericht

```csv
scan,zone,max_deviation_mm,tolerance_mm,status
scan_000,OOO,0.000,0.0,OK
scan_000,GRAT_XXX,0.847,1.2,OK
scan_000,ANGUSS_A,1.521,1.4,NOK
scan_000,ANGUSS_B,0.763,1.0,OK
scan_000,SCHLITZ_E,12.731,12.4-13.0,OK
```

### `inspection_rework_points.csv` – Rework-Koordinaten

```csv
scan,zone,x_mm,y_mm,z_mm,deviation_mm
scan_000,ANGUSS_A,12.543,-8.211,34.102,1.521
scan_000,ANGUSS_A,12.601,-8.189,34.087,1.489
...
```

> Die Rework-Datei enthält **ausschließlich** die Koordinaten von Punkten, die Toleranzgrenzen überschreiten – bereit für die direkte Übergabe an einen Roboter-Nacharbeitsprozess.

---

##  Bekannte Limitationen

### 1. Reflexionen auf metallischen Oberflächen
Die Intel RealSense D405 nutzt Infrarot-Stereo-Projektion. Stark glänzende oder spiegelnde Oberflächen können zu Lücken oder Rauschen in der Tiefenkarte führen, was zu False-Positive-Fehlern führen kann.

**Workaround:** Mattierungssprays für Testmessungen oder zukünftig Structured-Light-Sensoren einsetzen.

### 2. Voxel-Quantisierung
Das Voxel-Downsampling (0.5 mm) ist für die GPU-Performance notwendig, führt jedoch zu einem Verlust hochfrequenter Oberflächendetails. Abweichungen < 0.5 mm in OOO-Zonen können nicht zuverlässig detektiert werden.

### 3. ArUco-Kalibrierungsgenauigkeit
Die Pose-Estimation aus dem 2D-Bild unterliegt Beleuchtungsschwankungen (Jitter). Der ICP-Algorithmus kompensiert dies weitgehend, eine mechanisch fixierte Kalibrierung würde jedoch stabilere Ergebnisse liefern.

---

##  Abhängigkeiten

```
pyrealsense2     # Intel RealSense SDK
open3d           # 3D-Punktwolken-Verarbeitung
numpy            # Numerische Berechnungen
opencv-python    # ArUco Marker-Erkennung
rembg            # KI-Hintergrundentfernung (ISNET)
pyserial         # Serielle Kommunikation (Drehteller)
```

Vollständige Liste: siehe `requirements.txt`

---

##  Technische Details

### Alignment-Algorithmus

Das System verwendet ein **zweistufiges Registrierungsverfahren**:

1. **Grobausrichtung (RANSAC):**
   - FPFH Feature Matching (Fast Point Feature Histograms)
   - Radius: `VOXEL_SIZE * 5`, max. 4.000.000 Iterationen
   - Parallel für Originalposition und 180°-Flip (Symmetrie-Logik)

2. **Feinausrichtung (ICP – Point-to-Plane):**
   - GPU-beschleunigt via Open3D Tensor API
   - Minimiert: $E(R,t) = \sum_{i=1}^{N} ||(R \cdot p_i + t) - q_i||^2$
   - Konvergenz-Kriterium: max. 50 Iterationen, Toleranz 1e-6

### KI-Segmentierung

- **Modell:** ISNET-general-use (über rembg-Bibliothek)
- **Eingabe:** Hochauflösendes RGB-Bild der D405
- **Ausgabe:** Binäre Alpha-Maske des Bauteils
- **Vorteil:** Kein Parallaxenfehler (D405 hat keinen separaten RGB-Sensor)

---

##  Kontext & Zitation

Dieses Projekt entstand im Rahmen des Kurses **„KI-Anwendungen im betrieblichen Umfeld"** an der **Hochschule Niederrhein** (Wintersemester 2025/26).

```bibtex
@misc{ki_projekt_2026,
  title     = {Entwicklung eines automatisierten 360°-Prüfstands zur optischen Qualitätskontrolle 
               mittels Intel RealSense D405 und CAD-basiertem Soll-Ist-Abgleich},
  author    = {Kennerknecht, Lukas and Damek, Jonas},
  year      = {2026},
  school    = {Hochschule Niederrhein – University of Applied Sciences},
  note      = {KI-Anwendungen im betrieblichen Umfeld, WS 2025/26},
  url       = {https://github.com/lukas32123/AI_Segmentation_D405}
}
```

---

##  Lizenz

Dieses Projekt ist unter der [MIT-Lizenz](LICENSE) veröffentlicht.

---

<div align="center">

**Hochschule Niederrhein – University of Applied Sciences**  
Wintersemester 2025/26 · KI-Anwendungen im betrieblichen Umfeld

*Lukas Kennerknecht · Jonas Damek*

</div>
