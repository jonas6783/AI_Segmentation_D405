import pyrealsense2 as rs
import numpy as np
import time


class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        self.align = rs.align(rs.stream.color)

        
        self.temp_filter = rs.temporal_filter()
        self.threshold = rs.threshold_filter(0.01, 2.0)

        self.depth_scale_factor = 1000.0

    def start(self):
        profile = self.pipeline.start(self.config)

        # --- HIGH DENSITY PRESET ---
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 4)  
            print("[Camera] High Density Modus aktiviert.")

        scale = depth_sensor.get_depth_scale()
        self.depth_scale_factor = 1.0 / scale
  
        for _ in range(15):
            self.pipeline.wait_for_frames()

    def stop(self):
        self.pipeline.stop()

    def get_frame_data(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return None

        # Filter
        depth_frame = self.threshold.process(depth_frame)
        depth_frame = self.temp_filter.process(depth_frame)

        intrinsics = aligned.get_profile().as_video_stream_profile().get_intrinsics()

        return {
            'color': np.asanyarray(color_frame.get_data()),
            'depth': np.asanyarray(depth_frame.get_data()),
            'intrinsics': intrinsics,
            'scale_factor': self.depth_scale_factor
        }
