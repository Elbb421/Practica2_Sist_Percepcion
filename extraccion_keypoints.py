import open3d as o3d
import numpy as np
import copy

# Cargar nubes de puntos 
# ESCENA
escena = o3d.io.read_point_cloud("snap_0point.pcd") 
# OBJETOS
objeto1 = o3d.io.read_point_cloud("s0_mug_corr.pcd") 
objeto2 = o3d.io.read_point_cloud("s0_piggybank_corr.pcd") 
objeto3 = o3d.io.read_point_cloud("s0_plant_corr.pcd") 
objeto4 = o3d.io.read_point_cloud("s0_plc_corr.pcd") 

# Pintar el objeto para diferenciarlo (opcional)
# objeto.paint_uniform_color([1, 0.706, 0]) 

# Eliminar planos dominantes
def limpiar_escena(pcd):
    # Segmentación del plano dominante (suelo/mesa)
    # distance_threshold: margen para considerar un punto dentro del plano
    # ransac_n: puntos para estimar el plano
    # num_iterations: intentos del algoritmo
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000) [cite: 12]
    
    # Extraer los puntos que NO pertenecen al plano (outliers)
    escena_sin_plano = pcd.select_by_index(inliers, invert=True)
    return escena_sin_plano

escena_limpia = limpiar_escena(escena)