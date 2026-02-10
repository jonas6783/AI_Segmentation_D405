import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs


class Calibrator:
    def __init__(self, width, height, fps):
        self.width = width
        self.height = height
        self.fps = fps
        # Marker Dictionary (Hier 7x7 für deinen Marker)
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        except AttributeError:
            print("Fehler: OpenCV Version zu alt oder ArUco fehlt.")

    def find_marker_position(self, marker_size):
        """Gibt die X,Y,Z Position des Markers relativ zur Kamera zurück"""
        print(f"\n[Calibration] Suche Marker ({marker_size * 100:.1f} cm) als Nullpunkt...")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        profile = pipeline.start(config)

        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        camera_matrix = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs, dtype=np.float32)

        # 3D Punkte des Markers
        half = marker_size / 2.0
        obj_points = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
                              dtype=np.float32)

        final_tvec = None
        buffer_tvec = []

        try:
            for i in range(40):  
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame: continue

                img = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                corners, ids, _ = self.detector.detectMarkers(gray)

                if ids is not None and len(ids) > 0:
                    _, rvec, tvec = cv2.solvePnP(obj_points, corners[0], camera_matrix, dist_coeffs)

                    # tvec ist die Position [x, y, z]
                    buffer_tvec.append(tvec.reshape(3))  

                    cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                cv2.imshow("Kalibrierung (Nullpunkt)", img)
                cv2.waitKey(100)

            if len(buffer_tvec) > 10:
                # Durchschnitt berechnen (Mittelwert aller Messungen)
                final_tvec = np.mean(buffer_tvec, axis=0)
                print(f"[Calibration] Nullpunkt gefunden bei: {final_tvec}")
            else:
                print("[Calibration] Marker nicht stabil gefunden!")

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

        return final_tvec  # Gibt [x, y, z] zurück
