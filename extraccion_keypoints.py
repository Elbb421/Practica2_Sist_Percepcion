import open3d as o3d
import numpy as np

voxel_size = 0.001

# Archivos de objetos
nombres_objetos = [
    "s0_mug_corr.pcd",
    "s0_piggybank_corr.pcd",
    "s0_plant_corr.pcd",
    "s0_plc_corr.pcd"
]

# Colores opcionales
colores_objetos = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0]
]

# -----------------------------
# 1. Funciones auxiliares
# -----------------------------

def limpiar_escena(pcd):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    return pcd.select_by_index(inliers, invert=True)

def extraer_keypoints(pcd, voxel_size):
    return o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=5 * voxel_size,
        non_max_radius=5 * voxel_size,
        gamma_21=0.975,
        gamma_32=0.975
    )

def calcular_descriptores(pcd, keypoints, voxel_size):
    radius_normal = voxel_size * 2
    keypoints.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    return o3d.pipelines.registration.compute_fpfh_feature(
        keypoints,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

def encontrar_correspondencias(desc_obj, desc_sce):
    matcher = o3d.geometry.KDTreeFlann(desc_sce)
    correspondencias = []

    for i in range(desc_obj.data.shape[1]):
        [k, idx, dist] = matcher.search_knn_vector_xd(desc_obj.data[:, i], 1)
        correspondencias.append([i, idx[0]])

    return np.array(correspondencias)

# -----------------------------
# 2. Procesar escena
# -----------------------------

escena = o3d.io.read_point_cloud("snap_0point.pcd")
escena_limpia = limpiar_escena(escena)
escena_filtro = escena_limpia.voxel_down_sample(voxel_size)
kp_escena = extraer_keypoints(escena_filtro, voxel_size)
desc_escena = calcular_descriptores(escena_filtro, kp_escena, voxel_size)

print(f"Keypoints escena: {len(kp_escena.points)}")

# -----------------------------
# 3. Procesar todos los objetos en un bucle
# -----------------------------

objetos = []
keypoints_obj = []
descriptores_obj = []
matches_obj = []

for nombre in nombres_objetos:
    obj = o3d.io.read_point_cloud(nombre)
    obj_f = obj.voxel_down_sample(voxel_size)
    kp = extraer_keypoints(obj_f, voxel_size)
    desc = calcular_descriptores(obj_f, kp, voxel_size)
    matches = encontrar_correspondencias(desc, desc_escena)

    objetos.append(obj_f)
    keypoints_obj.append(kp)
    descriptores_obj.append(desc)
    matches_obj.append(matches)

    print(f"{nombre}: {len(kp.points)} keypoints, {len(matches)} matches")

# -----------------------------
# 4. Visualización
# -----------------------------

o3d.visualization.draw_geometries([escena_filtro, kp_escena])
