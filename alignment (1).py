import open3d as o3d
import numpy as np

# =================================================
# SETUP
# =================================================
VOXEL_SIZE = 1.5

DEVICE = (
    o3d.core.Device("CUDA:0")
    if o3d.core.cuda.is_available()
    else o3d.core.Device("CPU:0")
)

SCAN_COLORS = [
    [1.0, 0.0, 1.0], # Grün
   [1.0, 0.0, 1.0],
]

# =================================================
# HELFER
# =================================================
def get_pcd_gpu(path, is_scan=False):
    pcd = (
        o3d.io.read_triangle_mesh(path).sample_points_uniformly(100000)
        if (path.endswith(".ply") and not is_scan)
        else o3d.io.read_point_cloud(path)
    )

    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    if is_scan:
        pcd.scale(1000, center=(0, 0, 0))
        pcd, _ = pcd.remove_statistical_outlier(30, 1.0)

    pcd_gpu = o3d.t.geometry.PointCloud.from_legacy(pcd, device=DEVICE)
    pcd_gpu = pcd_gpu.voxel_down_sample(VOXEL_SIZE)
    pcd_gpu.estimate_normals(max_nn=30, radius=VOXEL_SIZE * 3)

    return pcd_gpu


def align_gpu(source, target):
    src_l = source.to_legacy()
    tgt_l = target.to_legacy()

    s_feat = o3d.pipelines.registration.compute_fpfh_feature(
        src_l,
        o3d.geometry.KDTreeSearchParamHybrid(VOXEL_SIZE * 5, 100),
    )
    t_feat = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_l,
        o3d.geometry.KDTreeSearchParamHybrid(VOXEL_SIZE * 5, 100),
    )

    res_g = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_l,
        tgt_l,
        s_feat,
        t_feat,
        True,
        VOXEL_SIZE * 3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                VOXEL_SIZE * 3
            )
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
    )

    res_f = o3d.t.pipelines.registration.icp(
        source,
        target,
        VOXEL_SIZE * 2,
        o3d.core.Tensor(res_g.transformation, o3d.core.float64, DEVICE),
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(
            1e-6, 1e-6, 50
        ),
    )

    return res_f


# =================================================
# HAUPTFUNKTION
# =================================================
def align_scans(cad_file, scan_files):
    cad_gpu = get_pcd_gpu(cad_file)
    cad_legacy = cad_gpu.to_legacy()
    cad_legacy.paint_uniform_color([0, 0, 0])

    aligned_scans = []
    metrics = []

    for i, file in enumerate(scan_files):
        print(f"Aligning {file} ...")

        s_orig = get_pcd_gpu(file, is_scan=True)

        res_a = align_gpu(s_orig, cad_gpu)

        R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi))
        R_t = o3d.core.Tensor(R, o3d.core.float64, DEVICE)
        s_flip = s_orig.clone().rotate(R_t, s_orig.get_center())

        res_b = align_gpu(s_flip, cad_gpu)

        if res_a.fitness >= res_b.fitness:
            res_final = res_a
            final = s_orig.transform(res_a.transformation)
        else:
            res_final = res_b
            final = s_flip.transform(res_b.transformation)

        scan_legacy = final.to_legacy()
        scan_legacy.paint_uniform_color(
            SCAN_COLORS[i % len(SCAN_COLORS)]
        )

        aligned_scans.append(scan_legacy)

        metrics.append(
            {
                "scan": file,
                "fitness": float(res_final.fitness),
                "rmse": float(res_final.inlier_rmse),
            }
        )

    return cad_legacy, aligned_scans, metrics

