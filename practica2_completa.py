"""
Pasos:
1. Eliminar planos dominantes (3 iteraciones de RANSAC plano)
2. Voxel Downsampling
3. ISS Keypoints (con fallback automático)
4. FPFH Descriptores
5. Correspondencias (ratio test de Lowe)
6. Registro global RANSAC
7. Refinamiento ICP

Estrategia de FPFH:
- Se intenta FPFH sobre ISS keypoints primero 
- Si hay pocos matches (< 3), se recurre a FPFH sobre la nube completa
  downsampled 

=====================================================================
"""

import open3d as o3d
import numpy as np
import copy
import time


#  CONFIGURACIÓN PCDs
# Rutas a los archivos 
RUTA_ESCENA = "snap_0point.pcd"

RUTAS_OBJETOS = [
    "s0_mug_corr.pcd",
    "s0_piggybank_corr.pcd",
    "s0_plant_corr.pcd",
    "s0_plc_corr.pcd",
]

NOMBRES_OBJETOS = ["Taza", "Hucha", "Planta", "PLC"]

# Colores RGB (0-1) para cada objeto en la visualización
COLORES_OBJETOS = [
    [1.0, 0.2, 0.2],   # Rojo
    [0.2, 0.8, 0.2],   # Verde
    [0.2, 0.4, 1.0],   # Azul
    [1.0, 0.8, 0.0],   # Amarillo
]

# =====================================================================
#  PARÁMETROS DEL PIPELINE
# =====================================================================
# Todos los radios son múltiplos de voxel_size. 
# Si se cambia voxel_size, todo se ajusta automáticamente.
#
# voxel_size:
# voxel_size > 0.009, no se pueden calcular normales,se baja.
# voxel_size muy pequeño, no hay descriptores suficientes
# =====================================================================

voxel_size = 0.003

# Segmentación de planos (RANSAC) 
# Número de planos dominantes a eliminar : 2 paredes y 1 mesa
NUM_PLANOS_ELIMINAR = 3

# distance_threshold: umbral de distancia al plano para ser inlier
# Si no se elimina bien plano → se sube a 0.015 o 0.02
# Si se borra parte de los objetos → se baja a 0.005 o 0.003
plane_distance_threshold = 0.01
plane_num_iterations = 1000

# ISS Keypoints 
#   gamma BAJO (0.3-0.5) → MÁS selectivo → MENOS keypoints (esquinas fuertes)
#   gamma ALTO (0.8-0.9) → MENOS selectivo → MÁS keypoints (cualquier punto)
iss_gamma_21 = 0.5
iss_gamma_32 = 0.5

# Radios ISS 
# salient_radius: radio para la matriz de dispersión, contexto amplio
# non_max_radius: radio para supresión de no-máximos, separación entre KP
iss_salient_radius = 10 * voxel_size   # 0.05
iss_non_max_radius  = 5  * voxel_size   # 0.025

# Normales y FPFH 
normal_radius = 4 * voxel_size   # Radio para estimar normales (≥ voxel_size)
fpfh_radius   = 8 * voxel_size   # Radio para FPFH, MÁS CRÍTICO

# Ratio test de Lowe para correspondencias
# 0.7: muy estricto (pocos matches, muy fiables)
# 0.8: estándar
# 0.9: permisivo (más matches, algunos erróneos)
ratio_threshold = 0.95

# RANSAC Registration 
# max_correspondence_distance: distancia máxima para considerar un par
# como inlier en RANSAC. Debe ser generoso para converger.
# - Demasiado pequeño → RANSAC no encuentra transformación
# - Demasiado grande → transformación errónea con 8 tmb va bien
# 3-5 veces voxel_size
ransac_max_distance = 5 * voxel_size 

# ransac_n: número mínimo de correspondencias para estimar transformación
# 3: mínimo absoluto para transformación rígida 3D (6 DOF)
# 4 más robusto
ransac_n = 4

# Número máximo de iteraciones de RANSAC
# Más iteraciones → más probabilidad de encontrar la transformación correcta
# pero más lento. 100 000 
ransac_max_iterations = 100000
ransac_confidence = 500  # Número de validaciones

# ICP 
# max_correspondence_distance: distancia máxima para correspondencias ICP
# Debe ser MÁS PEQUEÑO que RANSAC (refinamiento fino)
# 0.5-1.5 veces voxel_size funciona pero con 5 tmb
icp_max_distance = voxel_size * 1.5  # 0.0045

# Número máximo de iteraciones ICP
icp_max_iterations = 50


# 1: Eliminar planos dominantes 

def eliminar_planos(pcd, num_planos=3, distance_threshold=0.01,
                    num_iterations=1000):
    """
    Elimina múltiples planos dominantes iterativamente.
    Cada iteración:
    1. Encuentra el plano con más inliers (RANSAC)
    inliers son los puntos que pertenecen al plano detectado.
    2. Elimina los puntos del plano
    3. Repite con la nube restante

    """
    pcd_actual = pcd # escena que se va limpiando
    planos_encontrados = 0
    
    for i in range(num_planos):
        if len(pcd_actual.points) < 100:
            print(f"  [Plano {i+1}] Pocos puntos restantes ({len(pcd_actual.points)}), parando")
            break
        
        plane_model, inliers = pcd_actual.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations
        )
        
        [a, b, c, d] = plane_model
        n_puntos = len(pcd_actual.points)
        porcentaje = 100 * len(inliers) / max(n_puntos, 1)
        
        # Si el plano tiene muy pocos inliers, probablemente no es un plano real
        if porcentaje < 5:
            print(f"  [Plano {i+1}] Solo {porcentaje:.1f}% inliers, no es plano dominante. Parando.")
            break
        
        print(f"  [Plano {i+1}] {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
        print(f"           Inliers: {len(inliers)} ({porcentaje:.1f}%)")
        
        pcd_actual = pcd_actual.select_by_index(inliers, invert=True)
        planos_encontrados += 1
    
    print(f"  Planos eliminados: {planos_encontrados}")
    print(f"  Puntos restantes: {len(pcd_actual.points)}")
    return pcd_actual


#  2 Voxel Downsampling

def hacer_downsample(pcd, vs):
    """
    Reduce la densidad agrupando puntos en voxels cúbicos.
    Cada voxel se reemplaza por su centroide.
    
    ¿Por qué es necesario?
    - Una escena de Kinect puede tener 300,000+ puntos → muy lento
    - Con voxel_size=0.005 se reduce a ~10,000-20,000
    - Se mantiene la forma geométrica esencial
    """
    n_orig = len(pcd.points)
    pcd_down = pcd.voxel_down_sample(voxel_size=vs)
    n_down = len(pcd_down.points)
    pct = 100 * (1 - n_down / max(n_orig, 1))
    print(f"  Downsample (voxel={vs:.4f}m): {n_orig} → {n_down} puntos ({pct:.1f}% reducción)")
    return pcd_down

#  3: ISS Keypoints (con fallback)
def extraer_keypoints_iss(pcd, salient_radius, non_max_radius,
                          gamma_21, gamma_32, etiqueta=""):
    """
    Detecta keypoints usando Intrinsic Shape Signatures (ISS).
    
    ¿Qué hace ISS?
    1. Para cada punto, calcula la matriz de dispersión de su vecindad
    2. Obtiene los 3 eigenvalues: λ1 ≥ λ2 ≥ λ3
    3. Si λ2/λ1 < γ21 Y λ3/λ2 < γ32 → el punto es "interesante"
       (los 3 eigenvalues son diferentes → no es plano ni arista simple)
    4. Aplica Non-Maximum Suppression (NMS): en cada vecindario
       de radio non_max_radius, solo se queda el punto con mayor
       "saliency" (= λ3, el eigenvalue más pequeño)
    
    Interpretación de eigenvalues:
    - Plano:   λ1 >> λ2 ≈ λ3 ≈ 0  → ratios ≈ 0 y indefinido → NO es KP
    - Arista:  λ1 ≈ λ2 >> λ3 ≈ 0  → ratio1 alto, ratio2 bajo  → DEPENDE
    - Esquina: λ1 ≈ λ2 ≈ λ3       → ratios ≈ 1               → NO con γ bajo
    - Esquina suave: λ1 > λ2 > λ3  → ratios < 1               → SÍ es KP ✓
    
    ISS busca puntos donde los 3 eigenvalues sean DISTINTOS entre sí,
    no donde sean iguales. Son esquinas "suaves" y transiciones de curvatura.
    """
    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=gamma_21,
        gamma_32=gamma_32
    )
    n = len(keypoints.points)
    total = len(pcd.points)
    print(f"  ISS [{etiqueta}]: {n}/{total} keypoints ({100*n/max(total,1):.2f}%)")
    return keypoints


def extraer_keypoints_con_fallback(pcd, etiqueta=""):
    # Intento 1: parámetros estándar
    kp = extraer_keypoints_iss(
        pcd,
        salient_radius=iss_salient_radius,
        non_max_radius=iss_non_max_radius,
        gamma_21=iss_gamma_21,
        gamma_32=iss_gamma_32,
        etiqueta=f"{etiqueta} (estándar)"
    )
    if len(kp.points) > 0:
        return kp
    
    # Intento 2: gamma relajado
    print(f"    0 keypoints → reintentando con gamma relajado (0.75)...")
    kp = extraer_keypoints_iss(
        pcd,
        salient_radius=15 * voxel_size,
        non_max_radius=iss_non_max_radius,
        gamma_21=0.75,
        gamma_32=0.75,
        etiqueta=f"{etiqueta} (relajado)"
    )
    if len(kp.points) > 0:
        return kp
    
    # Intento 3: muy relajado
    print(f"    Todavía 0 → último intento (gamma=0.9, radio mayor)...")
    kp = extraer_keypoints_iss(
        pcd,
        salient_radius=20 * voxel_size,
        non_max_radius=iss_non_max_radius * 2,
        gamma_21=0.9,
        gamma_32=0.9,
        
        etiqueta=f"{etiqueta} (muy relajado)"
    )
    
    if len(kp.points) == 0:
        print(f"      Se usará FPFH sobre nube completa como fallback")
    
    return kp


# =====================================================================
#  FUNCIÓN 4: FPFH Descriptores
# =====================================================================

def calcular_fpfh(pcd, normal_r, fpfh_r, etiqueta=""):
    """
    Calcula descriptores Fast Point Feature Histograms (FPFH).
    
    Proceso:
    1. Estima normales por PCA local (eigenvector de menor eigenvalue)
    2. Calcula SPFH: histograma simplificado (3 angulos × 11 bins = 33D)
       para cada par punto-vecino directo
    3. FPFH = SPFH(p) + Σ w_i · SPFH(vecino_i)
       ponderado por distancia inversa
    
    IMPORTANTE: Normales deben apuntar en dirección consistente.
    Si las normales del objeto y la escena apuntan en direcciones
    opuestas, los descriptores serán completamente diferentes
    → 0 matches → pipeline falla.
    
    Usa orient_normals_towards_camera_location() si es necesario.
    """
    # Estimar normales
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_r, max_nn=30
        )
    )
    
    # Calcular FPFH
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=fpfh_r, max_nn=100
        )
    )
    dims, n_pts = fpfh.data.shape
    print(f"  FPFH [{etiqueta}]: {dims}D × {n_pts} puntos")
    return fpfh


# =====================================================================
#  FUNCIÓN 5: Correspondencias
# =====================================================================

def encontrar_correspondencias(desc_obj, desc_sce, ratio_thresh=0.9):
    """
    Emparejar descriptores del objeto con los de la escena.
    
    Usar KDTree para búsqueda eficiente + ratio test de Lowe:
    - Para cada descriptor del objeto, buscar los 2 más cercanos en escena
    - Solo aceptar si dist(mejor) < ratio_thresh × dist(segundo_mejor)
    
    ¿Por qué el ratio test?
    Si el mejor match está muy cerca del segundo mejor, significa que
    el descriptor es AMBIGUO (parecido a varios puntos de la escena).
    Esos matches probablemente son erróneos y deben descartarse.
    
    Ejemplo:
      Caso bueno: dist1=0.1, dist2=0.8 → ratio=0.125 < 0.9 → ACEPTAR 
      Caso dudoso: dist1=0.3, dist2=0.35 → ratio=0.86 < 0.9 → ACEPTAR (dudoso)
      Caso malo: dist1=0.3, dist2=0.31 → ratio=0.97 > 0.9 → RECHAZAR 
    """
    matcher = o3d.geometry.KDTreeFlann(desc_sce) # permite buscar vecinos más cercanos de forma muy rápida.
    correspondencias = []
    
    for i in range(desc_obj.data.shape[1]):
        [k, idx, dist_sq] = matcher.search_knn_vector_xd(
            desc_obj.data[:, i], 2
        )
        if k >= 2:
            d1 = np.sqrt(dist_sq[0])
            d2 = np.sqrt(dist_sq[1])
            if d1 < ratio_thresh * d2:
                correspondencias.append([i, idx[0]])
        elif k == 1:
            # Solo 1 vecino encontrado → aceptar sin ratio test
            # (esto pasa en nubes muy pequeñas)
            correspondencias.append([i, idx[0]])
    
    if len(correspondencias) == 0:
        return np.array([]).reshape(0, 2)
    return np.array(correspondencias)


# =====================================================================
#  FUNCIÓN 6: Registro global RANSAC
# =====================================================================

def registro_global_ransac(source, target, source_fpfh, target_fpfh,
                           max_distance, ransac_n=4, max_iter=100000,
                           confidence=500):
    """
    Registro global usando RANSAC basado en emparejamiento de features.
    
    ¿Qué hace?
    1. Itera muchas veces (hasta max_iter)
    2. Cada iteración: selecciona ransac_n correspondencias aleatorias,
       estima la transformación rígida, cuenta inliers
    3. Se queda con la transformación que tenga más inliers
    4. Los inliers son pares cuya distancia post-transformación < max_distance
    
    ¿Por qué se necesita mínimo 3 correspondencias?
    - Transformación rígida 3D = 6 grados de libertad (3 rot + 3 trasl)
    - Cada par da 3 ecuaciones → 2 pares = 6 ecuaciones (justo)
    - Pero 2 pares pueden ser degenerados (4 puntos coplanares)
    - 3 pares = 9 ecuaciones → sistema sobredeterminado → SVD funciona
    - 4 pares (ransac_n=4) → más robusto todavía
    
    Parámetros clave:
    - max_distance: umbral para considerar inlier. Debe ser GENEROSO
      porque la transformación inicial puede estar lejos.
      Regla: 3-5 × voxel_size
    - max_iter: más iteraciones = más probabilidad de éxito pero más lento
    """
    print(f"  RANSAC: max_dist={max_distance:.4f}, ransac_n={ransac_n}, "
          f"max_iter={max_iter}")
    
    # Checkers: condiciones adicionales para validar correspondencias
    distance_checker = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
        max_distance
    )
    edge_checker = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
        0.9  # Las aristas entre pares correspondientes deben tener
             # longitudes similares (ratio > 0.9)
    )
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target,
        source_fpfh, target_fpfh,
        mutual_filter=True,  # Solo correspondencias bidireccionales
        max_correspondence_distance=max_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[edge_checker, distance_checker],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iter, confidence
        )
    )
    
    print(f"  RANSAC resultado: fitness={result.fitness:.4f}, "
          f"RMSE={result.inlier_rmse:.4f}")
    return result


# =====================================================================
#  FUNCIÓN 7: Refinamiento ICP
# =====================================================================

def refinamiento_icp(source, target, init_transform, max_distance,
                     max_iterations=50):
    """
    Refinamiento local usando Iterative Closest Point (ICP).
    
    ¿Qué hace?
    Repite iterativamente:
    1. Aplica transformación actual al source
    2. Para cada punto del source transformado, busca el más cercano
       en el target
    3. Estima la transformación rígida que minimiza la distancia
       entre pares correspondientes
    4. Actualiza la transformación
    Hasta converger o alcanzar max_iterations
    
    ¿Por qué punto-a-plano?
    - Punto-a-punto: minimiza distancia entre puntos → convergencia lenta
    - Punto-a-plano: minimiza distancia a la superficie → convergencia
      más rápida y robusta
    - Requiere normales estimadas previamente
    
    Requisitos:
    - Necesita una BUENA aproximación inicial (de RANSAC)
    - Si la inicialización es mala, ICP se queda en un mínimo local
    - max_distance debe ser PEQUEÑO (refinamiento fino)
      Regla: 0.5-1.5 × voxel_size
    """
    print(f"  ICP: max_dist={max_distance:.4f}, max_iter={max_iterations}")
    
    # Calcular normales si no existen (necesarias para punto-a-plano)
    if not target.has_normals():
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )
    if not source.has_normals():
        source.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )
    
    result = o3d.pipelines.registration.registration_icp(
        source, target,
        max_correspondence_distance=max_distance,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        )
    )
    
    print(f"  ICP resultado: fitness={result.fitness:.4f}, "
          f"RMSE={result.inlier_rmse:.4f}")
    return result


# =====================================================================
#  FUNCIÓN DE PIPELINE PARA UN OBJETO
# =====================================================================

def procesar_objeto(nombre, ruta_obj, color, escena_down, desc_escena,
                    kp_escena, escena_limpia_original):
    """
    Pipeline completo para detectar un objeto en la escena.
    
    Estrategia:
    - Primero intentar FPFH sobre ISS keypoints
    - Si hay pocos matches, recurrir a FPFH sobre nube completa downsampled
    """
    print("\n" + "=" * 65)
    print(f"  PROCESANDO: {nombre} ({ruta_obj})")
    print("=" * 65)
    
    resultado = {
        'nombre': nombre,
        'color': color,
        'exito': False,
    }
    
    # Cargar objeto
    obj = o3d.io.read_point_cloud(ruta_obj)
    print(f"  Puntos originales: {len(obj.points)}")
    
    if len(obj.points) == 0:
        print(f"  ✗ Error: no se pudo cargar {ruta_obj}")
        return resultado
    
    # ── PASO 2: Downsample del objeto ──
    print(f"\n  [Paso 2] Voxel Downsampling")
    obj_down = hacer_downsample(obj, voxel_size)
    
    
    # ── PASO 3: ISS Keypoints ──
    print(f"\n  [Paso 3] ISS Keypoints")
    kp_obj = extraer_keypoints_con_fallback(obj_down, nombre)
    resultado['kp_obj'] = kp_obj
    resultado['obj_down'] = obj_down
    
    n_kp = len(kp_obj.points)
    
    # ── PASO 4 y 5: FPFH + Matching ──
    # Estrategia: intentar con keypoints, si falla usar nube completa
    
    matches = np.array([]).reshape(0, 2)
    source_for_ransac = None
    source_fpfh = None
    
    if n_kp >= 3:
        # Intento A: FPFH sobre ISS keypoints
        print(f"\n  [Paso 4a] FPFH sobre ISS keypoints ({n_kp} puntos)")
        
        # Calcular normales en keypoints
        kp_obj.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )
        
        # FPFH sobre keypoints — el radio debe ser MAYOR que para la
        # nube completa porque los keypoints están más separados
        fpfh_kp_radius = max(fpfh_radius * 2, 10 * voxel_size)
        fpfh_obj_kp = calcular_fpfh(kp_obj, normal_radius, fpfh_kp_radius,
                                     f"{nombre} KP")
        
        print(f"\n  [Paso 5a] Correspondencias (keypoints)")
        matches = encontrar_correspondencias(fpfh_obj_kp, desc_escena,
                                            ratio_threshold)
        print(f"  Matches (KP): {len(matches)}")
        
        if len(matches) >= 40:
            source_for_ransac = kp_obj
            source_fpfh = fpfh_obj_kp
            print(f"  ✓ Suficientes matches con keypoints")
    
    if len(matches) < 40:
        # Intento B: FPFH sobre nube completa downsampled
        print(f"\n  [Paso 4b] FPFH sobre nube completa downsampled")
        fpfh_obj_full = calcular_fpfh(obj_down, normal_radius, fpfh_radius,
                                       f"{nombre} completa")
        
        print(f"\n  [Paso 5b] Correspondencias (nube completa)")
        matches = encontrar_correspondencias(fpfh_obj_full, desc_escena,
                                            ratio_threshold)
        print(f"  Matches (completa): {len(matches)}")
        
        if len(matches) >= 3:
            source_for_ransac = obj_down
            source_fpfh = fpfh_obj_full
            print(f"  ✓ Suficientes matches con nube completa")
    
    resultado['matches'] = matches
    
    if len(matches) < 3:
        print(f"\n  ✗ INSUFICIENTES MATCHES ({len(matches)} < 3)")
        print(f"    RANSAC necesita mínimo 3 correspondencias no colineales")
        print(f"    Soluciones:")
        print(f"    - Aumentar fpfh_radius (probar 8-10 × voxel_size)")
        print(f"    - Subir ratio_threshold (probar 0.95)")
        print(f"    - Verificar que el objeto está en la escena")
        print(f"    - Reducir voxel_size para más puntos")
        return resultado
    
    # ── PASO 6: RANSAC Registration ──
    print(f"\n  [Paso 6] Registro global RANSAC")
    t0 = time.time()
    
    # Usar la escena downsampled como target
    result_ransac = registro_global_ransac(
        source_for_ransac, escena_down,
        source_fpfh, desc_escena,
        max_distance=ransac_max_distance,
        ransac_n=ransac_n,
        max_iter=ransac_max_iterations,
        confidence=ransac_confidence
    )
    t_ransac = time.time() - t0
    print(f"  Tiempo RANSAC: {t_ransac:.2f}s")
    
    resultado['transform_ransac'] = result_ransac.transformation
    resultado['fitness_ransac'] = result_ransac.fitness
    resultado['rmse_ransac'] = result_ransac.inlier_rmse
    
    if result_ransac.fitness < 0.01:
        print(f"  ⚠ RANSAC fitness muy bajo ({result_ransac.fitness:.4f})")
        print(f"    La transformación puede ser incorrecta")
    
    # ── PASO 7: ICP Refinement ──
    print(f"\n  [Paso 7] Refinamiento ICP")
    t0 = time.time()
    
    # ICP se aplica sobre las nubes ORIGINALES (sin downsample) para
    # máxima precisión, usando la transformación de RANSAC como inicio
    result_icp = refinamiento_icp(
        obj, escena_limpia_original,
        init_transform=result_ransac.transformation,
        max_distance=icp_max_distance,
        max_iterations=icp_max_iterations
    )
    t_icp = time.time() - t0
    print(f"  Tiempo ICP: {t_icp:.2f}s")
    
    resultado['transform_icp'] = result_icp.transformation
    resultado['fitness_icp'] = result_icp.fitness
    resultado['rmse_icp'] = result_icp.inlier_rmse
    resultado['exito'] = True
    
    # Aplicar transformación final al objeto original
    obj_registrado = copy.deepcopy(obj)
    obj_registrado.transform(result_icp.transformation)
    obj_registrado.paint_uniform_color(color)
    resultado['obj_registrado'] = obj_registrado
    
    print(f"\n  ✓ RESULTADO FINAL:")
    print(f"    Fitness ICP: {result_icp.fitness:.4f}")
    print(f"    RMSE ICP: {result_icp.inlier_rmse:.4f}")
    
    return resultado


# =====================================================================
#  VISUALIZACIÓN
# =====================================================================

def visualizar_escena_con_keypoints(escena_down, kp_escena):
    """Escena downsampled (gris) + Keypoints ISS (rojo grande)."""
    escena_vis = copy.deepcopy(escena_down)
    escena_vis.paint_uniform_color([0.7, 0.7, 0.7])
    
    geoms = [escena_vis]
    
    if len(kp_escena.points) > 0:
        kp_vis = copy.deepcopy(kp_escena)
        kp_vis.paint_uniform_color([1.0, 0.0, 0.0])
        geoms.append(kp_vis)
    
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Escena + {len(kp_escena.points)} Keypoints ISS (rojo)",
        width=1280, height=720
    )


def visualizar_objeto_con_keypoints(obj_down, kp_obj, nombre, color):
    """Objeto downsampled (gris) + Keypoints ISS (color)."""
    obj_vis = copy.deepcopy(obj_down)
    obj_vis.paint_uniform_color([0.7, 0.7, 0.7])
    
    geoms = [obj_vis]
    
    if len(kp_obj.points) > 0:
        kp_vis = copy.deepcopy(kp_obj)
        kp_vis.paint_uniform_color(color)
        geoms.append(kp_vis)
    
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"{nombre} + {len(kp_obj.points)} Keypoints ISS",
        width=800, height=600
    )


def visualizar_resultado_final(escena, resultados):
    """
    Visualización final: escena con todos los objetos registrados.
    
    Cada objeto se muestra en su color, posicionado donde el
    pipeline lo ha detectado.
    """
    # Escena en gris claro
    escena_vis = copy.deepcopy(escena)
    escena_vis.paint_uniform_color([0.85, 0.85, 0.85])
    
    geoms = [escena_vis]
    
    for r in resultados:
        if r['exito'] and 'obj_registrado' in r:
            geoms.append(r['obj_registrado'])
            print(f"  + {r['nombre']}: registrado (fitness={r['fitness_icp']:.4f})")
        else:
            print(f"  ✗ {r['nombre']}: no detectado")
    
    o3d.visualization.draw_geometries(
        geoms,
        window_name="RESULTADO FINAL: Objetos detectados en la escena",
        width=1280, height=720
    )


def visualizar_matches(escena_down, kp_escena, resultados):
    """Visualiza las líneas de correspondencia sobre la escena."""
    escena_vis = copy.deepcopy(escena_down)
    escena_vis.paint_uniform_color([0.7, 0.7, 0.7])
    
    geoms = [escena_vis]
    
    pts_sce = np.asarray(kp_escena.points) if len(kp_escena.points) > 0 else np.array([])
    
    for r in resultados:
        matches = r.get('matches', np.array([]).reshape(0, 2))
        kp_obj = r.get('kp_obj', o3d.geometry.PointCloud())
        color = r['color']
        nombre = r['nombre']
        
        if len(matches) == 0 or len(kp_obj.points) == 0 or len(pts_sce) == 0:
            continue
        
        pts_obj = np.asarray(kp_obj.points)
        lines = []
        colors_lines = []
        
        for m in matches[:200]:  # Limitar a 200 líneas para no saturar
            idx_obj, idx_sce = int(m[0]), int(m[1])
            if idx_obj < len(pts_obj) and idx_sce < len(pts_sce):
                lines.append([pts_obj[idx_obj], pts_sce[idx_sce]])
                colors_lines.append(color)
        
        if len(lines) > 0:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(
                [p for line in lines for p in line]
            )
            ls.lines = o3d.utility.Vector2iVector(
                [[i*2, i*2+1] for i in range(len(lines))]
            )
            ls.colors = o3d.utility.Vector3dVector(colors_lines)
            geoms.append(ls)
            print(f"  [{nombre}] {len(lines)} líneas")
    
    if len(geoms) > 1:
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Correspondencias en la escena",
            width=1280, height=720
        )


# =====================================================================
#  PIPELINE PRINCIPAL
# =====================================================================

def ejecutar_pipeline():
    print("=" * 65)
    print("  PIPELINE DE RECONOCIMIENTO 3D — 7 PASOS")
    print("=" * 65)
    
    # ── Cargar escena ──
    print("\n" + "─" * 50)
    print("  CARGANDO ESCENA")
    print("─" * 50)
    escena = o3d.io.read_point_cloud(RUTA_ESCENA)
    print(f"  Archivo: {RUTA_ESCENA}")
    print(f"  Puntos: {len(escena.points)}")
    
    # Mostrar dimensiones para ayudar a elegir voxel_size
    if len(escena.points) > 0:
        bbox = escena.get_axis_aligned_bounding_box()
        ext = bbox.get_extent()
        print(f"  Dimensiones: X={ext[0]:.3f}m, Y={ext[1]:.3f}m, Z={ext[2]:.3f}m")
        print(f"  Diagonal: {np.linalg.norm(ext):.3f}m")
        vs_recomendado = ext.min() / 300
        print(f"  Voxel recomendado: ~{vs_recomendado:.4f}m (actual: {voxel_size:.4f}m)")
    
    # ── PASO 1: Eliminar planos dominantes ──
    print("\n" + "─" * 50)
    print(f"  PASO 1: Eliminar {NUM_PLANOS_ELIMINAR} planos dominantes")
    print("─" * 50)
    escena_sin_planos = eliminar_planos(
        escena,
        num_planos=NUM_PLANOS_ELIMINAR,
        distance_threshold=plane_distance_threshold,
        num_iterations=plane_num_iterations
    )
    
    # Guardar copia para ICP (nube sin planos pero sin downsample)
    escena_limpia_original = copy.deepcopy(escena_sin_planos)
    
    # ── PASO 2: Downsample escena ──
    print("\n" + "─" * 50)
    print("  PASO 2: Voxel Downsampling (escena)")
    print("─" * 50)
    escena_down = hacer_downsample(escena_sin_planos, voxel_size)
    
    # ── PASO 3: ISS Keypoints (escena) ──
    print("\n" + "─" * 50)
    print("  PASO 3: ISS Keypoints (escena)")
    print("─" * 50)
    kp_escena = extraer_keypoints_con_fallback(escena_down, "ESCENA")
    
    # ── PASO 4: FPFH (escena) ──
    # NOTA: Calculamos FPFH sobre la nube COMPLETA downsampled
    # (no solo keypoints) para que el matching sea más robusto.
    # Los ISS keypoints se usan para visualización y como primer intento.
    print("\n" + "─" * 50)
    print("  PASO 4: FPFH sobre nube completa downsampled (escena)")
    print("─" * 50)
    desc_escena = calcular_fpfh(escena_down, normal_radius, fpfh_radius,
                                 "ESCENA")
    
    # ── Visualizar escena con keypoints ──
    print("\n  [VIS] Escena + Keypoints ISS")
    visualizar_escena_con_keypoints(escena_down, kp_escena)
    
    # ── Procesar cada objeto ──
    resultados = []
    
    for i, (ruta, nombre, color) in enumerate(
        zip(RUTAS_OBJETOS, NOMBRES_OBJETOS, COLORES_OBJETOS)
    ):
        result = procesar_objeto(
            nombre, ruta, color,
            escena_down, desc_escena, kp_escena,
            escena_limpia_original
        )
        resultados.append(result)
        
        # Visualizar keypoints del objeto
        if 'obj_down' in result and 'kp_obj' in result:
            visualizar_objeto_con_keypoints(
                result['obj_down'], result['kp_obj'],
                nombre, color
            )
    
    # ── Visualizar correspondencias ──
    print("\n  [VIS] Correspondencias")
    visualizar_matches(escena_down, kp_escena, resultados)
    
    # ── Visualizar resultado final ──
    print("\n  [VIS] RESULTADO FINAL")
    visualizar_resultado_final(escena_limpia_original, resultados)
    
    # ── RESUMEN ──
    print("\n" + "=" * 65)
    print("  RESUMEN DE RESULTADOS")
    print("=" * 65)
    print(f"{'Objeto':<10} {'KP obj':<7} {'KP esc':<7} "
          f"{'Matches':<8} {'RANSAC fit':<11} {'ICP fit':<9} {'ICP RMSE':<9} {'Detectado'}")
    print("─" * 75)
    for r in resultados:
        n_kp = len(r.get('kp_obj', o3d.geometry.PointCloud()).points)
        n_m = len(r.get('matches', np.array([]).reshape(0, 2)))
        r_fit = r.get('fitness_ransac', 0)
        i_fit = r.get('fitness_icp', 0)
        i_rmse = r.get('rmse_icp', 0)
        detect = "✓ SÍ" if r['exito'] else "✗ NO"
        print(f"{r['nombre']:<10} {n_kp:<7} {len(kp_escena.points):<7} "
              f"{n_m:<8} {r_fit:<11.4f} {i_fit:<9.4f} {i_rmse:<9.4f} {detect}")
    
    return resultados


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == "__main__":
    resultados = ejecutar_pipeline()
