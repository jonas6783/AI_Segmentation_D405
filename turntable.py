# turntable.py
import serial
import time


class Turntable:
    def __init__(self, port, baudrate):
        self.ser = None
        try:
            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)
            print(f"[Turntable] Verbunden an {port}")
        except Exception as e:
            print(f"[Turntable] FEHLER: Konnte nicht verbinden ({e}). Simulation aktiv.")

    def rotate(self, degrees, speed=800):
        if self.ser:
            command = f"G91\nG1 X{degrees} F{speed}\n"
            self.ser.write(command.encode())

        # Wartezeit berechnen
        duration = (abs(degrees) / speed * 60) + 1.5
        time.sleep(duration)
        print(f"[Turntable] Gedreht um {degrees}°")

    def close(self):
        if self.ser:
            self.ser.close()