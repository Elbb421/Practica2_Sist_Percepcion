import open3d as o3d
import numpy as np

# ============================================================
# CONFIGURACIÓN
# ============================================================

voxel_size = 0.001

nombres_objetos = [
    "s0_mug_corr.pcd",
    "s0_piggybank_corr.pcd",
    "s0_plant_corr.pcd",
    "s0_plc_corr.pcd"
]

# ============================================================
# FUNCIÓN 1: Eliminar plano dominante (RANSAC)
# ============================================================

def limpiar_escena(pcd):
    print("\n[1] Eliminando plano dominante (RANSAC)...")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    pcd_clean = pcd.select_by_index(inliers, invert=True)
    print(f"  Puntos: {len(pcd.points)} → {len(pcd_clean.points)} tras eliminar plano")
    return pcd_clean

# ============================================================
# FUNCIÓN 2: Downsampling con estadísticas
# ============================================================

def hacer_downsample(pcd, voxel_size):
    n_inicial = len(pcd.points)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    n_final = len(pcd_down.points)
    reduccion = 100 * (1 - n_final / max(n_inicial, 1))

    print(f"  [Downsample] voxel={voxel_size:.4f} m | "
          f"{n_inicial} → {n_final} puntos | reducción: {reduccion:.1f}%")

    return pcd_down

# ============================================================
# FUNCIÓN 3: Keypoints ISS
# ============================================================

def extraer_keypoints(pcd, voxel_size):
    print("  Extrayendo keypoints ISS...")
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=5 * voxel_size,
        non_max_radius=5 * voxel_size,
        gamma_21=0.975,
        gamma_32=0.975
    )
    print(f"    Keypoints detectados: {len(keypoints.points)}")
    return keypoints

# ============================================================
# FUNCIÓN 4: FPFH sobre nube filtrada (NO sobre keypoints)
# ============================================================

def calcular_descriptores(pcd, voxel_size):
    print("  Calculando FPFH...")
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    print(f"    FPFH calculado: {fpfh.data.shape[1]} descriptores")
    return fpfh

# ============================================================
# FUNCIÓN 5: Matching con KDTree
# ============================================================

def encontrar_correspondencias(desc_obj, desc_sce):
    print("  Buscando correspondencias...")
    matcher = o3d.geometry.KDTreeFlann(desc_sce)
    correspondencias = []

    for i in range(desc_obj.data.shape[1]):
        [k, idx, dist] = matcher.search_knn_vector_xd(desc_obj.data[:, i], 1)
        correspondencias.append([i, idx[0]])

    print(f"    Total correspondencias: {len(correspondencias)}")
    return np.array(correspondencias)

# ============================================================
# PROCESAR ESCENA
# ============================================================

print("\n==============================")
print(" PROCESANDO ESCENA PRINCIPAL ")
print("==============================")

escena = o3d.io.read_point_cloud("snap_0point.pcd")
escena_limpia = limpiar_escena(escena)
escena_filtro = hacer_downsample(escena_limpia, voxel_size)
kp_escena = extraer_keypoints(escena_filtro, voxel_size)
desc_escena = calcular_descriptores(escena_filtro, voxel_size)

# ============================================================
# PROCESAR OBJETOS EN BUCLE
# ============================================================

print("\n==============================")
print(" PROCESANDO OBJETOS ")
print("==============================")

objetos_filtrados = []
keypoints_obj = []
descriptores_obj = []
matches_obj = []

for nombre in nombres_objetos:
    print(f"\n--- Objeto: {nombre} ---")

    obj = o3d.io.read_point_cloud(nombre)
    obj_f = hacer_downsample(obj, voxel_size)
    kp = extraer_keypoints(obj_f, voxel_size)
    desc = calcular_descriptores(obj_f, voxel_size)
    matches = encontrar_correspondencias(desc, desc_escena)

    objetos_filtrados.append(obj_f)
    keypoints_obj.append(kp)
    descriptores_obj.append(desc)
    matches_obj.append(matches)

# ============================================================
# VISUALIZACIÓN
# ============================================================

print("\nMostrando escena con keypoints...")
kp_escena.paint_uniform_color([1, 0, 0])  # rojo
o3d.visualization.draw_geometries([escena_filtro, kp_escena])
