"""
test_visualizacion.py
=====================================================================
Script rápido para comprobar que tus archivos PCD se cargan y
visualizan correctamente. Úsalo ANTES de ejecutar el pipeline.

Ejecuta: python test_visualizacion.py
=====================================================================
"""

import open3d as o3d
import numpy as np

# ===== TUS ARCHIVOS =====
RUTA_ESCENA = "snap_0point.pcd"
RUTAS_OBJETOS = [
    "s0_mug_corr.pcd",
    "s0_piggybank_corr.pcd",
    "s0_plant_corr.pcd",
    "s0_plc_corr.pcd",
]
NOMBRES = ["Taza", "Hucha", "Planta", "PLC"]
COLORES = [[1,0,0], [0,1,0], [0,0,1], [1,1,0]]

# ===== CARGAR Y MOSTRAR INFO =====
print("=" * 55)
print("  TEST DE VISUALIZACIÓN DE PCDs")
print("=" * 55)

# Escena
print(f"\n--- Escena: {RUTA_ESCENA} ---")
escena = o3d.io.read_point_cloud(RUTA_ESCENA)
print(f"  Puntos: {len(escena.points)}")
if len(escena.points) > 0:
    bbox = escena.get_axis_aligned_bounding_box()
    ext = bbox.get_extent()
    center = bbox.get_center()
    print(f"  Dimensiones: X={ext[0]:.3f}m, Y={ext[1]:.3f}m, Z={ext[2]:.3f}m")
    print(f"  Centro: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Tiene normales: {escena.has_normals()}")
    print(f"  Tiene colores: {escena.has_colors()}")
    
    # Calcular voxel_size recomendado
    vs_min = ext.min() / 400
    vs_max = ext.min() / 150
    print(f"  Voxel_size recomendado: entre {vs_min:.4f} y {vs_max:.4f}")
    
    # Mostrar escena
    escena_vis = o3d.geometry.PointCloud(escena)
    if not escena_vis.has_colors():
        escena_vis.paint_uniform_color([0.7, 0.7, 0.7])
    
    print(f"\n  → Mostrando escena (cierra la ventana para continuar)")
    o3d.visualization.draw_geometries(
        [escena_vis],
        window_name=f"ESCENA: {len(escena.points)} puntos",
        width=1280, height=720
    )
else:
    print("  ✗ ERROR: No se pudo cargar la escena")
    print(f"    ¿Existe el archivo {RUTA_ESCENA}?")

# Objetos
for ruta, nombre, color in zip(RUTAS_OBJETOS, NOMBRES, COLORES):
    print(f"\n--- Objeto: {nombre} ({ruta}) ---")
    obj = o3d.io.read_point_cloud(ruta)
    print(f"  Puntos: {len(obj.points)}")
    
    if len(obj.points) > 0:
        bbox = obj.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        print(f"  Dimensiones: X={ext[0]:.3f}m, Y={ext[1]:.3f}m, Z={ext[2]:.3f}m")
        print(f"  Tiene normales: {obj.has_normals()}")
        print(f"  Tiene colores: {obj.has_colors()}")
        
        # Downsample rápido para verificar que funciona
        vs_test = 0.005
        obj_down = obj.voxel_down_sample(vs_test)
        print(f"  Downsample (v={vs_test}): {len(obj.points)} → {len(obj_down.points)} puntos")
        
        # Probar normales
        try:
            obj_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            print(f"  Normales: OK ({len(obj_down.points)} normales estimadas)")
        except Exception as e:
            print(f"  Normales: ERROR → {e}")
            print(f"    → Prueba voxel_size más pequeño")
        
        # Probar ISS
        try:
            kp = o3d.geometry.keypoint.compute_iss_keypoints(
                obj_down,
                salient_radius=0.05,
                non_max_radius=0.025,
                gamma_21=0.5,
                gamma_32=0.5
            )
            print(f"  ISS Keypoints: {len(kp.points)}")
            
            if len(kp.points) == 0:
                # Reintentar con gamma relajado
                kp = o3d.geometry.keypoint.compute_iss_keypoints(
                    obj_down,
                    salient_radius=0.075,
                    non_max_radius=0.025,
                    gamma_21=0.75,
                    gamma_32=0.75
                )
                print(f"  ISS (relajado): {len(kp.points)}")
        except Exception as e:
            print(f"  ISS: ERROR → {e}")
            kp = o3d.geometry.PointCloud()
        
        # Visualizar
        obj_vis = o3d.geometry.PointCloud(obj)
        obj_vis.paint_uniform_color([0.7, 0.7, 0.7])
        geoms = [obj_vis]
        
        if len(kp.points) > 0:
            kp_vis = o3d.geometry.PointCloud(kp)
            kp_vis.paint_uniform_color(color)
            geoms.append(kp_vis)
        
        print(f"  → Mostrando {nombre} (cierra para continuar)")
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"{nombre}: {len(obj.points)} pts, {len(kp.points)} KP",
            width=800, height=600
        )
    else:
        print(f"  ✗ ERROR: No se pudo cargar {ruta}")

# Prueba de segmentación de planos
if len(escena.points) > 0:
    print(f"\n--- Prueba de segmentación de planos ---")
    pcd_test = o3d.geometry.PointCloud(escena)
    
    for i in range(3):
        if len(pcd_test.points) < 100:
            print(f"  [Plano {i+1}] Pocos puntos, parando")
            break
        
        plane_model, inliers = pcd_test.segment_plane(
            distance_threshold=0.01, ransac_n=3, num_iterations=1000
        )
        [a, b, c, d] = plane_model
        pct = 100 * len(inliers) / len(pcd_test.points)
        print(f"  [Plano {i+1}] {a:.3f}x+{b:.3f}y+{c:.3f}z+{d:.3f}=0 | "
              f"{len(inliers)} inliers ({pct:.1f}%)")
        
        pcd_test = pcd_test.select_by_index(inliers, invert=True)
    
    print(f"  Puntos tras eliminar planos: {len(pcd_test.points)}")

print("\n" + "=" * 55)
print("  TEST COMPLETADO")
print("=" * 55)