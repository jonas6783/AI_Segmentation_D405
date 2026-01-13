import open3d as o3d
import numpy as np
import copy
import time

class GeometryInspector:
    def __init__(self, voxel_size=0.003):
        self.voxel_size = voxel_size

    def preprocess_point_cloud(self, pcd):
        """
        Bereitet Wolke vor. 
        KORREKTUR: Erzwingt Ausrichtung zur Kamera, um 180-Grad-Fehler zu beheben.
        """
        # 1. Downsample
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        
        # 2. Normalen berechnen
        radius_normal = self.voxel_size * 3.0 
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # WICHTIG: Zurück zu "Towards Camera". 
        # "Tangent Plane" hat wahrscheinlich deine Normalen nach innen gedreht,
        # weshalb der Algorithmus das Bauteil auf den Kopf stellen musste, damit es passt.
        # [0,0,0] ist standardmäßig der Ursprung/Kamera bei Scans.
        pcd_down.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

        # 3. Features berechnen (FPFH)
        # Radius groß lassen (12x), das war gut gegen Z-Rotation
        radius_feature = self.voxel_size * 12.0
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=150))
        return pcd_down, pcd_fpfh

    def register_blindly(self, scan_raw, cad_raw, attempts=40, min_fitness_threshold=0.65):
        print(f"   [Inspector] Starte robuste Suche (Voxel: {self.voxel_size*1000:.1f}mm)...")
        
        # 1. Vorbereitung & Zentrierung
        source = copy.deepcopy(scan_raw)
        target = copy.deepcopy(cad_raw)
        
        source_center = source.get_center()
        target_center = target.get_center()
        source.translate(-source_center)
        target.translate(-target_center)

        # Preprocessing (Mit korrigierten Normalen!)
        source_down, source_fpfh = self.preprocess_point_cloud(source)
        target_down, target_fpfh = self.preprocess_point_cloud(target)

        # RANSAC Setup
        distance_threshold = self.voxel_size * 1.5
        
        checkers = [
            # Kantenlänge
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.70),
            
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            
            # Normalen-Check: 45 Grad ist sicher.
            # Da die Normalen jetzt garantiert zur Kamera zeigen, 
            # wird RANSAC gezwungen, Vorderseite auf Vorderseite zu legen.
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.deg2rad(45))
        ]
        
        best_result = None
        best_fitness = -1.0

        # 2. Der Wettbewerb (40 Versuche reichen meistens)
        for i in range(attempts):
            print(f"      -> Versuch {i+1}/{attempts} (Bester Fit: {best_fitness:.2f})...", end="\r")
            
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                True, distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, 
                checkers, 
                o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999)
            )
            
            # Grober Check mit Standard ICP
            icp_check = o3d.pipelines.registration.registration_icp(
                source_down, target_down, self.voxel_size * 2.0, ransac.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )

            if icp_check.fitness > best_fitness:
                best_fitness = icp_check.fitness
                best_result = icp_check
            
            if best_fitness > min_fitness_threshold:
                print(f"\n      -> Treffer! Versuch {i+1} war gut genug.")
                break
        
        print(f"\n   [Inspector] Beste Position gefunden (Fit: {best_fitness:.4f})")

        if best_result is None or best_fitness < 0.1:
            return None

        # --- 3. FINALER FEINSCHLIFF ---
        # Point-to-Plane ICP
        fine_threshold = self.voxel_size * 1.0 

        # Normalen Update (Auch hier zur Kamera orientieren!)
        radius_normal = self.voxel_size * 2.0
        
        if not source.has_normals(): 
            source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            source.orient_normals_towards_camera_location([0,0,0])
            
        if not target.has_normals(): 
            target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            # CAD ist geschlossen, hier ist tangent_plane oder default okay, aber wir lassen es so.

        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, 
            fine_threshold, 
            best_result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60)
        )

        # --- Transform zurückrechnen ---
        T_found = icp_fine.transformation
        M1 = np.eye(4); M1[:3, 3] = -source_center
        M3 = np.eye(4); M3[:3, 3] = target_center
        final_transform = M3 @ T_found @ M1
        
        final_res = o3d.pipelines.registration.RegistrationResult()
        final_res.fitness = icp_fine.fitness
        final_res.inlier_rmse = icp_fine.inlier_rmse
        final_res.transformation = final_transform
        
        return final_res
