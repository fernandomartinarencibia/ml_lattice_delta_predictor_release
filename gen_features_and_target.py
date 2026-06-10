from fpylll import IntegerMatrix, GSO, LLL
import pandas as pd
import numpy as np
import time
from multiprocessing import Pool, cpu_count
from lattice_utils import (
    GENERATORS, np_to_fpylll,
    profile_mass, profile_entropy, profile_adjacency
)

DELTAS = [0.55, 0.75, 0.90, 0.99] #deltas para reducción LLL
DATASET_SEED = 42  # semilla maestra para generación determinista del dataset

# d >= 120 es computacionalmente más costoso; 1000 muestras × 4 deltas son suficientes
SAMPLES_PER_DIM = {10: 2000, 20: 2000, 30: 2000, 50: 2000, 80: 2000,
                   100: 1000, 120: 1000, 150: 1000, 200: 1000}


def extract_features(B):
    M = GSO.Mat(B)
    M.update_gso()
    d = B.nrows

    B_np = np.array([list(B[i]) for i in range(d)], dtype=float)

    # M.get_log_det devuelve logaritmo del determinante AL CUADRADO, por eso se divide /2.0
    log_det = M.get_log_det(0, d) / 2.0

    norms_b = np.array([B[i].norm() for i in range(d)])

    # M.get_r devuelve EL CUADRADO de la norma GSO, raiz cuadrada para obtener ||b_i^*||
    norms_b_star = np.sqrt(np.array([M.get_r(i, i) for i in range(d)]))

    log_norms_b_star = np.log(norms_b_star)
    indices = np.arange(d)
    slope, intercept = np.polyfit(indices, log_norms_b_star, 1)
    residuals = log_norms_b_star - (slope * indices + intercept)

    log_orth_defect = np.sum(np.log(norms_b)) - log_det

    Gram = B_np.dot(B_np.T)
    off_diag_mask = ~np.eye(d, dtype=bool)
    off_diag_elements = np.abs(Gram[off_diag_mask])

    cond_num = np.linalg.cond(B_np)
    log_cond_num = np.log(cond_num)

    # Se extraen para j < i (triángulo inferior) usando M.get_mu(i,j)
    mu_vals = np.array([abs(M.get_mu(i, j)) for i in range(1, d) for j in range(i)])

    std_B = np.std(B_np)

    features = {
        'd': d,
        'log_det': log_det,

        # Estadisticas ||b_i||
        'norm_mean': np.mean(norms_b),
        'norm_std': np.std(norms_b),
        'norm_max': np.max(norms_b),
        'norm_min': np.min(norms_b),
        'norm_ratio': np.max(norms_b) / np.min(norms_b) if np.min(norms_b) > 0 else 0,

        # Estadisticas ||b_i^*||
        'gso_mean': np.mean(norms_b_star),
        'gso_std': np.std(norms_b_star),
        'gso_max': np.max(norms_b_star),
        'gso_min': np.min(norms_b_star),
        'gso_ratio': np.max(norms_b_star) / np.min(norms_b_star) if np.min(norms_b_star) > 0 else 0,
        'gso_slope': slope,
        'gso_res_std': np.std(residuals),

        # Defecto ortogonalidad y Matriz Gram
        'log_orth_defect': log_orth_defect,
        'gram_off_diag_mean': np.mean(off_diag_elements),
        'gram_off_diag_max': np.max(off_diag_elements),

        # Condicion, mu y B
        'log_cond_num': log_cond_num,
        'mu_mean': np.mean(mu_vals),
        'mu_max': np.max(mu_vals),
        'mu_frac_gt_05': np.sum(mu_vals > 0.5) / len(mu_vals),
        'std_B': std_B,

        # Funcionales del perfil GSO (forma global del perfil, útil para q-arios)
        'profile_mass': profile_mass(log_norms_b_star),
        'profile_entropy': profile_entropy(log_norms_b_star),
        'profile_adjacency': profile_adjacency(log_norms_b_star),
    }

    return features


def extract_targets(B_reduced):
    d = B_reduced.nrows

    M = GSO.Mat(B_reduced)
    M.update_gso()

    log_det = M.get_log_det(0, d) / 2.0

    norm_b1_red = B_reduced[0].norm()
    log_norm_b1_red = np.log(norm_b1_red)

    sum_log_norms_red = sum([np.log(B_reduced[i].norm()) for i in range(d)])
    log_orth_defect_red = sum_log_norms_red - log_det

    # RHF: ( ||b_1|| / |det(B)|^(1/d) )^(1/d)
    # Aplicando logaritmos: log(RHF) = (1/d) * [ log(||b_1||) - (1/d) * log_det ]
    log_rhf = (log_norm_b1_red - (log_det / d)) / d
    rhf = np.exp(log_rhf)

    return {
        'target_log_orth_defect': log_orth_defect_red,
        'target_log_norm_b1': log_norm_b1_red,
        'target_rhf': rhf
    }


def process_single_lattice(args):
    """Genera UNA matriz del tipo y dimensión dados y la reduce con TODOS los deltas."""
    d, lattice_type, seed = args
    np.random.seed(seed)  # cada worker recibe su propia semilla derivada de DATASET_SEED
    gen_func = GENERATORS[lattice_type]
    try:
        B_np = gen_func(d)
    except ValueError:
        return []

    B = np_to_fpylll(B_np)
    features = extract_features(B)
    features['lattice_type'] = lattice_type

    results = []
    for delta in DELTAS:
        row_data = features.copy()
        B_copy = IntegerMatrix(B)
        LLL.reduction(B_copy, delta=delta)
        targets = extract_targets(B_copy)
        row_data.update(targets)
        row_data['delta'] = delta
        results.append(row_data)

    return results


if __name__ == "__main__":
    inicio = time.time()

    dimensions = [10, 20, 30, 50, 80, 100, 120, 150, 200]

    # Generar semillas únicas por tarea a partir de DATASET_SEED para reproducibilidad
    rng = np.random.default_rng(DATASET_SEED)
    tasks = [
        (d, lt, int(rng.integers(0, 2**31)))
        for lt in GENERATORS
        for d in dimensions
        for _ in range(SAMPLES_PER_DIM[d])
    ]

    print(f"Iniciando procesamiento con {cpu_count()} núcleos...")
    print(f"Total tareas: {len(tasks)} matrices × {len(DELTAS)} deltas = {len(tasks) * len(DELTAS)} filas esperadas")

    with Pool(processes=cpu_count()) as pool:
        nested_dataset = pool.map(process_single_lattice, tasks)

    dataset = [row for sublist in nested_dataset for row in sublist]

    df = pd.DataFrame(dataset)
    df.to_csv("lattice_dataset_v2.csv", index=False)
    print(f"Dataset guardado con {len(df)} filas.")
    fin = time.time()
    print(f"Tiempo total de ejecución: {fin - inicio:.1f} segundos.")
