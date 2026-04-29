import open3d as o3d
import numpy as np

voxel_size = 0.005

nombres_objetos = [
    "s0_mug_corr.pcd",
    "s0_piggybank_corr.pcd",
    "s0_plant_corr.pcd",
    "s0_plc_corr.pcd"
]

colores_objetos = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
]

# ============================================================
# 1. Limpieza de escena
# ============================================================

def limpiar_escena(pcd):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    return pcd.select_by_index(inliers, invert=True)

# ============================================================
# 2. Downsample
# ============================================================

def hacer_downsample(pcd, voxel_size):
    return pcd.voxel_down_sample(voxel_size)

# ============================================================
# 3. Keypoints ISS
# ============================================================

def extraer_keypoints(pcd, voxel_size):
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=5 * voxel_size,
        non_max_radius=5 * voxel_size,
        gamma_21=0.5,
        gamma_32=0.5
    )
    return keypoints

# ============================================================
# 4. FPFH sobre keypoints
# ============================================================

def calcular_fpfh_keypoints(keypoints, voxel_size):
    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh

# ============================================================
# 5. Registro global (RANSAC)
# ============================================================

def registro_global_keypoints(obj_kp, escena_kp, fpfh_obj, fpfh_escena, voxel_size):
    distancia_max = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        obj_kp, escena_kp,
        fpfh_obj, fpfh_escena,
        mutual_filter=True,
        max_correspondence_distance=distancia_max,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distancia_max)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500)
    )
    return result

# ============================================================
# 6. Refinamiento ICP
# ============================================================

def refinar_icp(obj_original, escena_original, transform_inicial, voxel_size):

    # Calcular normales una sola vez
    radius_normal = voxel_size * 2
    obj_original.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    escena_original.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    distancia_max = voxel_size * 0.5

    result_icp = o3d.pipelines.registration.registration_icp(
        obj_original, escena_original,
        max_correspondence_distance=distancia_max,
        init=transform_inicial,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return result_icp

# ============================================================
# 7. Pipeline completo para un objeto
# ============================================================

def procesar_objeto(nombre_objeto, escena_limpia, color):

    obj = o3d.io.read_point_cloud(nombre_objeto)
    obj.paint_uniform_color(color)

    # Downsample
    obj_down = hacer_downsample(obj, voxel_size)
    escena_down = hacer_downsample(escena_limpia, voxel_size)

    # Normales en nubes densas (una sola vez)
    radius_normal = voxel_size * 2
    obj_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    escena_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # Keypoints
    kp_obj = extraer_keypoints(obj_down, voxel_size)
    kp_escena = extraer_keypoints(escena_down, voxel_size)

    # FPFH en keypoints
    fpfh_obj = calcular_fpfh_keypoints(kp_obj, voxel_size)
    fpfh_escena = calcular_fpfh_keypoints(kp_escena, voxel_size)

    # Registro global
    result_global = registro_global_keypoints(kp_obj, kp_escena, fpfh_obj, fpfh_escena, voxel_size)

    # Refinamiento ICP
    result_icp = refinar_icp(obj, escena_limpia, result_global.transformation, voxel_size)

    # Aplicar transformación refinada
    obj_transformado = obj.transform(result_icp.transformation)

    return obj_transformado

# ============================================================
# 8. Procesar escena y objetos
# ============================================================

escena = o3d.io.read_point_cloud("snap_0point.pcd")
escena_limpia = limpiar_escena(escena)

geoms = [escena_limpia]

for nombre, color in zip(nombres_objetos, colores_objetos):
    obj_transformado = procesar_objeto(nombre, escena_limpia, color)
    geoms.append(obj_transformado)

o3d.visualization.draw_geometries(geoms)
