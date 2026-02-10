import open3d as o3d
import numpy as np
import os
from rembg import remove, new_session
import config


class PointCloudProcessor:
    def __init__(self, marker_position_vector):
        
        self.marker_pos = marker_position_vector

        self.save_folder = config.OUTPUT_FOLDER
        os.makedirs(self.save_folder, exist_ok=True)

        if config.VERBOSE:
            print("[Processor] MODE:")
            print(" KI → Marker → Rotation → Box → (DBSCAN) → (Outlier)")
            print(f" KI            : {'AN' if config.USE_AI_MASK else 'AUS'}")
            print(f" DBSCAN        : {'AN' if config.USE_DBSCAN else 'AUS'}")
            print(f" Outlier       : {'AN' if config.USE_OUTLIER_REMOVAL else 'AUS'}")
            print(" Marker = Mittelpunkt der Box")
            print(f"[Processor] Output: '{self.save_folder}/'")

        self.session = None
        if config.USE_AI_MASK:
            print("[Processor] Lade KI Modell (ISNET)...")
            self.session = new_session("isnet-general-use")

        # -------- BOX: Marker = Mittelpunkt --------
        self.box_min = np.array([
            -config.BOX_HALF_X,
            -config.BOX_HALF_Y,
            -config.BOX_HALF_Z
        ])

        self.box_max = np.array([
             config.BOX_HALF_X,
             config.BOX_HALF_Y,
             config.BOX_HALF_Z
        ])



    # --------------------------------------------------
    # CLEANING PIPELINE
    # --------------------------------------------------
    def clean_point_cloud(self, pcd):
        if len(pcd.points) < 100:
            return pcd

        # ---- DBSCAN ----
        if config.USE_DBSCAN:
            labels = np.array(
                pcd.cluster_dbscan(
                    eps=config.DBSCAN_EPS,
                    min_points=config.DBSCAN_MIN_POINTS,
                    print_progress=False
                )
            )

            if labels.size > 0 and labels.max() >= 0:
                counts = np.bincount(labels[labels >= 0])
                largest_cluster = np.argmax(counts)
                indices = np.where(labels == largest_cluster)[0]
                pcd = pcd.select_by_index(indices)

        # ---- Outlier Removal ----
        if config.USE_OUTLIER_REMOVAL:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=config.OUTLIER_NB_NEIGHBORS,
                std_ratio=config.OUTLIER_STD_RATIO
            )

        return pcd

    # --------------------------------------------------
    # FRAME VERARBEITUNG
    # --------------------------------------------------
    def process_frame(self, data, angle):
        if config.VERBOSE:
            print(f"[Processor] Winkel {angle}° verarbeiten...")

        color_raw = data['color']
        depth_raw = data['depth']

        # ---------- 1. KI-HINTERGRUND (OPTIONAL) ----------
        if config.USE_AI_MASK and self.session is not None:
            try:
                result = remove(color_raw, session=self.session, alpha_matting=False)
                mask = result[:, :, 3] > 0

                clean_depth = depth_raw.copy()
                clean_depth[~mask] = 0
            except Exception as e:
                print(f"[Processor] KI Fehler → Fallback ohne KI: {e}")
                clean_depth = depth_raw
        else:
            clean_depth = depth_raw

        # ---------- 2. RGBD → Punktwolke ----------
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_raw),
            o3d.geometry.Image(clean_depth),
            depth_scale=data['scale_factor'],
            depth_trunc=config.DEPTH_TRUNC,
            convert_rgb_to_intensity=False
        )

        intr = data['intrinsics']
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            intr.width, intr.height,
            intr.fx, intr.fy,
            intr.ppx, intr.ppy
        )

        source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # ---------- 3. MARKER-ZENTRIERUNG ----------
        if self.marker_pos is not None:
            source.translate(-self.marker_pos)
        else:
            source.translate((0, 0, -0.15))

        # ---------- 4. ROTATION ----------
        rad = -angle * np.pi / 180.0
        R = source.get_rotation_matrix_from_xyz((0, rad, 0))
        source.rotate(R, center=(0, 0, 0))

        # ---------- 5. BOX (Marker = Mittelpunkt) ----------
        aabb = o3d.geometry.AxisAlignedBoundingBox(
            self.box_min,
            self.box_max
        )
        source = source.crop(aabb)

# ---------- 5.5 VOXEL DOWNSAMPLING (Fusion der Ebenen) ----------
        source = source.voxel_down_sample(
            voxel_size=0.0005  # 0.5 mm
        )

        # ---------- 6. CLEANING (OPTIONAL) ----------
        source = self.clean_point_cloud(source)

        # ---------- 7. SPEICHERN ----------
        filename = f"{self.save_folder}/scan_{int(angle):03d}.ply"
        o3d.io.write_point_cloud(filename, source)





