"""
Genera particiones train/val/test SIN fuga de pacientes a partir de mammo-bench.csv.

Problema que resuelve
---------------------
Los archivos train.csv / val.csv / test.csv actuales fueron partidos por imagen,
no por paciente. Como cada paciente aporta ~2.5 imagenes (CC/MLO, izquierda/derecha),
el 92% de las filas de test pertenecen a pacientes que el modelo ya vio en train.
Eso infla las metricas: el modelo memoriza el pecho, no aprende la patologia.

Uso
---
    python make_splits.py --bench mammo-bench.csv --outdir splits

    # variantes utiles
    python make_splits.py --keep-rsna              # conserva rsna-screening (100% benigno)
    python make_splits.py --include-suspicious     # 'Suspicious Malignant' -> malignant
    python make_splits.py --holdout-source cmmd    # evaluacion cruzada: test = solo cmmd
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# rsna-screening aporta 2265 imagenes y TODAS son benignas. Un modelo que solo
# reconozca "esto viene de rsna" acierta esas 2265 sin mirar la lesion. Se excluye
# por defecto; usa --keep-rsna si quieres conservarlo.
DEFAULT_EXCLUDE = ["rsna-screening"]

LABEL_MAP = {"benign": 0, "malignant": 1}


def build_patient_key(df):
    """Clave unica de paciente. Los IDs se repiten entre datasets, hay que prefijar."""
    sid = df["source_subjectid"].astype(str).str.strip()
    # Filas sin ID de sujeto: se tratan como pacientes individuales para no
    # agruparlas todas juntas bajo la clave 'nan'.
    missing = sid.isin(["nan", "", "None"])
    sid = sid.mask(missing, "row" + df.index.astype(str))
    return df["source_dataset"].astype(str) + "|" + sid


def load_bench(path, include_suspicious, exclude_sources):
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    cls = df["classification"].astype(str).str.strip().str.lower()
    if include_suspicious:
        cls = cls.replace({"suspicious malignant": "malignant"})

    df = df[cls.isin(LABEL_MAP)].copy()
    df["label"] = cls[cls.isin(LABEL_MAP)].map(LABEL_MAP).values

    if exclude_sources:
        before = len(df)
        df = df[~df["source_dataset"].isin(exclude_sources)].copy()
        print(f"Excluidos {before - len(df)} registros de {exclude_sources}")

    df = df.reset_index(drop=True)
    df["patient_key"] = build_patient_key(df)
    return df


def grouped_split(df, test_frac, val_frac, seed):
    """
    Particion estratificada por (dataset, etiqueta) y agrupada por paciente.

    Estratificar por el par (dataset, etiqueta) -- y no solo por la etiqueta --
    mantiene la misma mezcla de fuentes en los tres subconjuntos, de modo que la
    diferencia entre val y test no sea simplemente un cambio de dominio.
    """
    strata = df["source_dataset"].astype(str) + "_" + df["label"].astype(str)
    groups = df["patient_key"].values

    n_splits_test = max(2, int(round(1.0 / test_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(sgkf.split(df, strata, groups))

    dev = df.iloc[dev_idx].reset_index(drop=True)
    # val_frac esta expresado sobre el total, hay que reescalarlo sobre dev
    val_frac_dev = val_frac / (1.0 - test_frac)
    n_splits_val = max(2, int(round(1.0 / val_frac_dev)))
    sgkf2 = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    dev_strata = dev["source_dataset"].astype(str) + "_" + dev["label"].astype(str)
    tr_idx, val_idx = next(sgkf2.split(dev, dev_strata, dev["patient_key"].values))

    return (
        dev.iloc[tr_idx].reset_index(drop=True),
        dev.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


def holdout_source_split(df, holdout, val_frac, seed):
    """Test = un dataset completo. Mide generalizacion real a un dominio nuevo."""
    test = df[df["source_dataset"] == holdout].reset_index(drop=True)
    if test.empty:
        raise SystemExit(f"El dataset '{holdout}' no existe en el CSV.")
    dev = df[df["source_dataset"] != holdout].reset_index(drop=True)

    strata = dev["source_dataset"].astype(str) + "_" + dev["label"].astype(str)
    n_splits = max(2, int(round(1.0 / val_frac)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, val_idx = next(sgkf.split(dev, strata, dev["patient_key"].values))
    return dev.iloc[tr_idx].reset_index(drop=True), dev.iloc[val_idx].reset_index(drop=True), test


def audit(name, split, train_keys=None):
    n = len(split)
    mal = split["label"].mean() * 100
    print(f"\n[{name}] {n} imagenes | {split['patient_key'].nunique()} pacientes | {mal:.1f}% maligno")
    print(pd.crosstab(split["source_dataset"], split["label"]).to_string())
    if train_keys is not None:
        leak = split["patient_key"].isin(train_keys).sum()
        print(f"  Pacientes compartidos con train: {leak} filas ({leak / max(n,1) * 100:.2f}%)")


def source_prior_baseline(train, test):
    """
    Cuanto se puede acertar mirando SOLO de que dataset viene la imagen.
    Es el suelo que hay que superar; si tu modelo no lo supera con holgura,
    esta aprendiendo el dominio, no la lesion.
    """
    prior = train.groupby("source_dataset")["label"].mean()
    pred = (test["source_dataset"].map(prior).fillna(0.5) > 0.5).astype(int)
    acc = (pred == test["label"]).mean() * 100
    print(f"\nBaseline 'solo identidad del dataset' sobre test: {acc:.2f}% accuracy")
    print("  (tu modelo tiene que superar esto por un margen amplio para valer algo)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="mammo-bench.csv")
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-rsna", action="store_true",
                    help="conserva rsna-screening (por defecto se excluye: 0%% de malignos)")
    ap.add_argument("--include-suspicious", action="store_true",
                    help="trata 'Suspicious Malignant' como maligno")
    ap.add_argument("--holdout-source", default=None,
                    help="usa un dataset completo como test (evaluacion cross-dataset)")
    args = ap.parse_args()

    exclude = [] if args.keep_rsna else DEFAULT_EXCLUDE
    df = load_bench(args.bench, args.include_suspicious, exclude)
    print(f"Total utilizable: {len(df)} imagenes de {df['patient_key'].nunique()} pacientes")

    if args.holdout_source:
        train, val, test = holdout_source_split(df, args.holdout_source, args.val_frac, args.seed)
    else:
        train, val, test = grouped_split(df, args.test_frac, args.val_frac, args.seed)

    train_keys = set(train["patient_key"])
    audit("train", train)
    audit("val", val, train_keys)
    audit("test", test, train_keys)
    source_prior_baseline(train, test)

    os.makedirs(args.outdir, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(args.outdir, f"{name}.csv")
        split.to_csv(path, index=False)
        print(f"Escrito {path}")


if __name__ == "__main__":
    main()
