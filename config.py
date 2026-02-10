# ==============================

CAMERA_WIDTH = 1280  #
CAMERA_HEIGHT = 720  #
CAMERA_FPS = 30
AUTOEXPOSURE_FRAMES = 15
# HARDWARE
# ==============================
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200


# ==============================
# MARKER
# ==============================
MARKER_SIZE = 0.025   # 25 mm


# ==============================
# SCAN
# ==============================
ROTATION_STEP = 20.0
TOTAL_ANGLE = 360.0


# ==============================
# BOX (Marker = Mittelpunkt!)
# ==============================
# Halbausdehnungen der Box (± um Marker)
BOX_HALF_X = 0.1  
BOX_HALF_Y = 0.1          
BOX_HALF_Z = 0.1          


# ==============================
# FILTER FLAGS
# ==============================
USE_AI_MASK = True          
USE_DBSCAN = False         
USE_OUTLIER_REMOVAL = False

# ==============================
# POINT CLOUD FILTERING
# ==============================

# --- DBSCAN ---
DBSCAN_EPS = 0.002          

DBSCAN_MIN_POINTS = 10  

# --- OUTLIER REMOVAL ---
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0



# ==============================
# DEPTH / SENSOR
# ==============================
DEPTH_TRUNC = 0.75   


# ==============================
# DEBUG
# ==============================
VERBOSE = True

# ==============================
# OUTPUT
# ==============================
OUTPUT_FOLDER = "scan_clean_pipeline"




