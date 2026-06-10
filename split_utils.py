import numpy as np
import pandas as pd
import os

CSV_PATH   = 'lattice_dataset_v2.csv'
SPLIT_PATH = 'splits/canonical_test_base_ids.csv'


def reconstruct_base_id(df: pd.DataFrame) -> pd.Series:
    """base_id canónico = 'd|lattice_type|occ', alineado con las filas de df.

    Requiere que df tenga las columnas 'd' y 'lattice_type' (antes de get_dummies).
    occ = cumcount() // 4 dentro de cada grupo (d, lattice_type), replicando
    exactamente el criterio histórico de dim_models / dim_mlps / oracle.
    """
    occ = df.groupby(['d', 'lattice_type']).cumcount() // 4
    base_id = (df['d'].astype(str) + '|'
               + df['lattice_type'].astype(str) + '|'
               + occ.astype(str))
    return base_id


def build_test_base_ids(df: pd.DataFrame, test_size: float = 0.2,
                        seed: int = 42) -> set:
    """Asigna el test_size de las bases a test, estratificado por (d, lattice_type).

    Garantiza que cada estrato quede representado proporcionalmente en test
    (evita estratos vacíos o con varianza ~0 sin datos de test).
    """
    base_id = reconstruct_base_id(df)
    bases = (pd.DataFrame({'base_id': base_id,
                           'd': df['d'].values,
                           'lt': df['lattice_type'].values})
             .drop_duplicates('base_id'))
    rng = np.random.default_rng(seed)
    test_ids = []
    for (_d, _lt), grp in bases.groupby(['d', 'lt'], sort=True):
        ids = rng.permutation(grp['base_id'].to_numpy())
        n_test = round(test_size * len(ids))
        test_ids.extend(ids[:n_test].tolist())
    return set(test_ids)


def materialize_split(test_size: float = 0.2, seed: int = 42) -> set:
    """Calcula y persiste la partición canónica en SPLIT_PATH. Idempotente."""
    df = pd.read_csv(CSV_PATH)
    test_ids = build_test_base_ids(df, test_size, seed)
    os.makedirs(os.path.dirname(SPLIT_PATH), exist_ok=True)
    pd.Series(sorted(test_ids), name='base_id').to_csv(SPLIT_PATH, index=False)
    print(f"Partición canónica guardada: {len(test_ids)} bases de test → {SPLIT_PATH}")
    return test_ids


def load_test_base_ids(path: str = SPLIT_PATH) -> set:
    """Carga la partición canónica; la crea si no existe."""
    if not os.path.exists(path):
        return materialize_split()
    return set(pd.read_csv(path)['base_id'])
