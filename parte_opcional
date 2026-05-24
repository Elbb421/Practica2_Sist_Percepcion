#Se sustituye la función icp en la practica2_completa.py
import numpy as np
import open3d as o3d

def icp_point_to_point_manual(source_pcd, target_pcd,
                              init_transform=np.eye(4),
                              max_iterations=50,
                              max_distance=0.01,
                              tolerance=1e-6):
    """
    ICP punto-a-punto implementado a bajo nivel.

    - source_pcd: PointCloud del objeto (Open3D)
    - target_pcd: PointCloud de la escena (Open3D)
    - init_transform: matriz 4x4 inicial (por ejemplo, salida de RANSAC)
    - max_iterations: máximo de iteraciones ICP
    - max_distance: distancia máxima para aceptar correspondencias
    - tolerance: umbral de cambio mínimo en el error para parar

    Devuelve:
    - transform: matriz 4x4 final
    - history: lista con el error medio por iteración
    """

    # Copia para no modificar el original
    source = copy.deepcopy(source_pcd)
    target = copy.deepcopy(target_pcd)

    # Aplicar transformación inicial
    source.transform(init_transform)

    # Convertir a arrays
    src_pts = np.asarray(source.points)
    tgt_pts = np.asarray(target.points)

    # KDTree sobre la escena (target)
    kdtree = o3d.geometry.KDTreeFlann(target)

    T = init_transform.copy()
    prev_error = np.inf
    history = []

    for it in range(max_iterations):
        correspondencias_src = []
        correspondencias_tgt = []

        # 1. Buscar correspondencias (vecino más cercano)
        for p in src_pts:
            [k, idx, dist_sq] = kdtree.search_knn_vector_3d(p, 1)
            if k == 1:
                d = np.sqrt(dist_sq[0])
                if d < max_distance:
                    correspondencias_src.append(p)
                    correspondencias_tgt.append(tgt_pts[idx[0]])

        if len(correspondencias_src) < 3:
            print(f"[ICP manual] Iter {it}: menos de 3 correspondencias válidas, paro.")
            break

        P = np.asarray(correspondencias_src)  # source
        Q = np.asarray(correspondencias_tgt)  # target

        # 2. Calcular centroides
        mu_P = P.mean(axis=0)
        mu_Q = Q.mean(axis=0)

        # 3. Centrar
        P_centered = P - mu_P
        Q_centered = Q - mu_Q

        # 4. Matriz de covarianza
        H = P_centered.T @ Q_centered

        # 5. SVD
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Corregir reflexión si det(R) < 0
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = mu_Q - R @ mu_P

        # 6. Actualizar transformación global T
        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = t

        T = T_step @ T

        # 7. Aplicar a los puntos source
        src_pts_h = np.hstack((src_pts, np.ones((src_pts.shape[0], 1))))
        src_pts = (T_step @ src_pts_h.T).T[:, :3]

        # 8. Calcular error medio
        errores = np.linalg.norm((R @ P.T).T + t - Q, axis=1)
        error_medio = errores.mean()
        history.append(error_medio)

        print(f"[ICP manual] Iter {it:02d}: error medio = {error_medio:.6f}, "
              f"corresp = {len(P)}")

        # 9. Criterio de parada
        if abs(prev_error - error_medio) < tolerance:
            print(f"[ICP manual] Convergencia alcanzada (Δerror < {tolerance})")
            break

        prev_error = error_medio

    return T, history

