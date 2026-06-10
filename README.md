# ML_lattice_delta_predictor

TFG sobre criptoanálisis de redes mediante Machine Learning. Predice propiedades de bases de retículos reducidas con LLL y construye un oráculo de selección adaptativa del parámetro delta.

---

## Cómo reproducir los experimentos

Ejecutar en este orden exacto desde la raíz del repositorio:

### 1. Crear el entorno conda

```bash
conda env create -f environment.yml
conda activate delta_predictor_release
```

### 2. Generar el dataset 

```bash
python gen_features_and_target.py
```

Genera `lattice_dataset_v2.csv` (~224 000 filas). Con `DATASET_SEED = 42` el resultado es determinista. Tiempo estimado: ~30 min en CPU con todos los núcleos.

### 3. Entrenar modelos XGBoost por dimensión

Abrir y ejecutar todas las celdas de **`dim_models.ipynb`**.

Produce:
- `models/per_dim/gb_*.pkl` y `scaler_*.pkl` (55 archivos)
- `models/per_dim/xgb_results.json` (métricas R²/MAE para comparativas)

Tiempo estimado: ~10 min con GPU (CUDA).

### 4. MLPs por dimensión

Abrir y ejecutar todas las celdas de **`dim_mlps.ipynb`**.

**Requiere:** paso 3 completado (`models/per_dim/xgb_results.json` debe existir).

Produce:
- `models/mlp_per_dim/{target}_d{d}.pt` (27 archivos)
- `graphs/mlp_all_targets.png`

Tiempo estimado: ~30 min con GPU.

### 5. Oráculo de selección de delta

Abrir y ejecutar todas las celdas de **`oracle.ipynb`**.

**Requiere:** paso 3 completado (`models/per_dim/gb_target_rhf_d*.pkl` deben existir).

Produce tablas de accuracy y speedup estratificadas por tipo de retículo y dimensión.

Tiempo estimado: ~5 min.

---

## Estructura del repositorio

```
ML_lattice_cryptanalysis/
├── gen_features_and_target.py   # Generación del dataset
├── lattice_utils.py             # Generadores de retículos y funcionales GSO
├── environment.yml              # Dependencias conda
├── dim_models.ipynb             # Modelos XGBoost por dimensión (PRINCIPAL)
├── oracle.ipynb                 # Oráculo de delta con modelos per-dim
├── dim_mlps.ipynb               # MLPs por dimensión (comparativa)
│
├── models/
│   ├── per_dim/                 # Modelos XGBoost serializados
│   └── mlp_per_dim/             # Modelos MLP por dimensión
│
├── graphs/                      # Figuras generadas
```

---

## Notas de reproducibilidad

- **Semilla del dataset:** `DATASET_SEED = 42` en `gen_features_and_target.py`. Cada worker recibe una semilla derivada → el CSV es completamente determinista.
- **Semilla de modelos:** todos los notebooks usan `SEED = 42` con `random_state=42` en splits y `np.random.seed()` / `torch.manual_seed()`.
- **GPU:** los modelos XGBoost usan `device='cuda'`. Si no hay GPU disponible, cambiar a `device='cpu'` en `dim_models.ipynb` y `compute_savings.py`.
- **Tolerancia de R²:** diferencias de ±0.01 entre ejecuciones son normales si se regenera el dataset.
