# main.py
import config
from turntable import Turntable
from camera1 import RealSenseCamera
from processor2 import PointCloudProcessor
from calibration import Calibrator
import numpy as np


def main():
    print("--- D405 Scanner: Marker-Mode ---")

    # Standard-Position (15cm geradeaus), falls Kalibrierung übersprungen wird
    marker_pos = np.array([0.0, 0.0, 0.15])

    choice = input("Marker-Kalibrierung starten? (j/n): ").lower()
    if choice == 'j':
        calib = Calibrator(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)
        found_pos = calib.find_marker_position(config.MARKER_SIZE)

        if found_pos is not None:
            marker_pos = found_pos
            print(f"--> Kalibrierung OK. Marker bei: {marker_pos}")
            print("Stelle jetzt das Objekt auf den Marker.")
            input("Drücke ENTER zum Starten...")
        else:
            print("Kalibrierung fehlgeschlagen! Nutze Standardwerte.")

    # Scanner Start
    turntable = Turntable(config.SERIAL_PORT, config.BAUDRATE)
    camera = RealSenseCamera(config.CAMERA_WIDTH, config.CAMERA_HEIGHT, config.CAMERA_FPS)

    
    processor = PointCloudProcessor(marker_pos)

    camera.start()
    current_angle = 0.0

    try:
        while current_angle < config.TOTAL_ANGLE:
            print(f"Scan {current_angle}°...")

            frame_data = camera.get_frame_data()
            processor.process_frame(frame_data, current_angle)

            if current_angle + config.ROTATION_STEP < config.TOTAL_ANGLE:
                turntable.rotate(config.ROTATION_STEP)

            current_angle += config.ROTATION_STEP

    finally:
        camera.stop()
        turntable.close()

print("Scan Vorgang Abgeschlossen")


if __name__ == "__main__":
    main()
