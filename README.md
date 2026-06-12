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

### 2. Obtener el dataset

#### Vía normal (recomendada): usar el dataset congelado

El dataset canónico que sustenta **todos los resultados reportados en la memoria** se distribuye congelado para garantizar la reproducibilidad exacta de las cifras. Basta con descomprimirlo:

```bash
unzip lattice_dataset_v2.zip
```

Se obtiene `lattice_dataset_v2.csv` (~224 000 filas, 30 columnas). Conviene verificar su integridad frente a la huella publicada:

```bash
sha256sum lattice_dataset_v2.csv
# esperado: d3e5a9d8a50ec188e6246703d72d20d548d5bfaeba067f6fc264eb75e48d949c
```

> En Windows (PowerShell): `Expand-Archive lattice_dataset_v2.zip .` y `Get-FileHash lattice_dataset_v2.csv -Algorithm SHA256`.

#### Vía alternativa: regenerar el dataset desde cero

```bash
python gen_features_and_target.py
```

Genera un `lattice_dataset_v2.csv` (~224 000 filas). Con `DATASET_SEED = 42` cada *worker* recibe una semilla derivada de forma determinista, por lo que **el fichero resultante es idéntico entre ejecuciones**, con independencia del número de núcleos o del orden de paralelización. Tiempo estimado: ~30 min en CPU con todos los núcleos.

> **Importante.** El dataset canónico (vía normal) se generó con anterioridad a la consolidación de este esquema de semillas por tarea. Por ello, la regeneración produce un conjunto **distinto, aunque estadísticamente equivalente**, del empleado en los experimentos. Para reproducir las cifras *exactas* de la memoria, usar la vía normal (el dataset congelado); la regeneración garantiza la reproducibilidad del *método*, no del fichero concreto, y puede arrojar variaciones perceptibles —no significativas— en las métricas.

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
├── lattice_dataset_v2.zip       # Dataset congelado (vía normal) — descomprimir antes del paso 3
├── gen_features_and_target.py   # Generación del dataset (vía alternativa)
├── lattice_utils.py             # Generadores de retículos y funcionales GSO
├── split_utils.py               # Partición canónica entrenamiento/prueba
├── environment.yml              # Dependencias conda
├── dim_models.ipynb             # Modelos XGBoost por dimensión (PRINCIPAL)
├── oracle.ipynb                 # Oráculo de delta con modelos per-dim
├── dim_mlps.ipynb               # MLPs por dimensión (comparativa)
│
├── models/
│   ├── per_dim/                 # Modelos XGBoost serializados
│   └── mlp_per_dim/             # Modelos MLP por dimensión
│
├── splits/
│   └── canonical_test_base_ids.csv  # Lista canónica de bases de test
│
├── graphs/                      # Figuras generadas
```

---

## Notas de reproducibilidad

- **Dataset canónico congelado:** los resultados de la memoria se obtienen sobre el `lattice_dataset_v2.csv` incluido en `lattice_dataset_v2.zip`. Se distribuye congelado (con su huella `SHA-256`) porque precede a la incorporación del esquema actual de semillas por tarea; usarlo es la única forma de reproducir las cifras exactas.
- **Semilla del dataset (regeneración):** `DATASET_SEED = 42` en `gen_features_and_target.py`. Un generador maestro `np.random.default_rng(42)` reparte una semilla por tarea, de modo que la regeneración es determinista entre ejecuciones (pero distinta del fichero congelado, véase el paso 2).
- **Partición canónica:** la asignación entrenamiento/prueba es única y se persiste en `splits/canonical_test_base_ids.csv`. Todos los notebooks la cargan mediante `load_test_base_ids()` (`split_utils.py`), garantizando que ninguna base se reparta entre entrenamiento y prueba.
- **Semilla de modelos:** todos los notebooks usan `SEED = 42`, con `random_state=42` en los splits y `np.random.seed()` / `torch.manual_seed()` en NumPy y PyTorch.
- **GPU:** los modelos XGBoost usan `device='cuda'`. Si no hay GPU disponible, cambiar a `device='cpu'` en `dim_models.ipynb` (y en cualquier notebook que use CUDA).
- **Tolerancia de R²:** sobre el dataset congelado, diferencias de ±0.01 en R² entre ejecuciones son normales y se deben al no determinismo residual de operaciones de punto flotante en GPU; no alteran las conclusiones. Si en su lugar se regenera el dataset, las diferencias pueden ser algo mayores por tratarse de un conjunto distinto, pero estadísticamente equivalente.
