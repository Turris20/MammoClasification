"""
Prueba de DeLong para comparar dos AUC sobre los MISMOS casos.

Por que hace falta
------------------
Comparar dos modelos mirando si sus intervalos de confianza individuales se
solapan es una prueba incorrecta y muy conservadora. Dos IC del 95% pueden
solaparse y aun asi la diferencia ser significativa, porque el IC de cada AUC
mide la incertidumbre de ESE AUC frente a un remuestreo independiente, no la
incertidumbre de la DIFERENCIA.

Ademas, aqui los dos modelos se evaluan sobre exactamente los mismos casos y
comparten backbone y datos de entrenamiento, asi que sus errores estan muy
correlacionados. La prueba de DeLong (1988), en la implementacion rapida de
Sun y Xu (2014), estima esa covarianza y la descuenta, lo que le da mucha mas
potencia que comparar intervalos por separado.

Uso:
    python article/delong_test.py --mil runs_mil/predicciones_test_mil.csv \
        --cnn runs/predicciones_test.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy import stats


def _midrank(x):
    """Rangos con promedio en los empates, necesarios para los componentes de DeLong."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_cov(y_true, scores):
    """
    Componentes estructurales de DeLong para k modelos evaluados en los mismos casos.

    Devuelve (aucs, matriz de covarianza k x k).
    """
    pos = scores[:, y_true == 1]
    neg = scores[:, y_true == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]

    tx = np.array([_midrank(pos[r]) for r in range(k)])
    ty = np.array([_midrank(neg[r]) for r in range(k)])
    tz = np.array([_midrank(np.concatenate([pos[r], neg[r]])) for r in range(k)])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    # v01: contribucion de cada positivo; v10: contribucion de cada negativo.
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    return aucs, np.cov(v01) / m + np.cov(v10) / n


def compare(y_true, score_a, score_b, name_a="A", name_b="B"):
    """Contrasta H0: AUC_a = AUC_b sobre los mismos casos."""
    aucs, cov = delong_cov(np.asarray(y_true), np.vstack([score_a, score_b]))
    cov = np.atleast_2d(cov)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var_diff <= 0:
        return dict(auc_a=aucs[0], auc_b=aucs[1], diff=diff, z=np.nan, p=np.nan,
                    ci=(np.nan, np.nan), r=np.nan)
    se = np.sqrt(var_diff)
    z = diff / se
    r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return dict(auc_a=aucs[0], auc_b=aucs[1], diff=diff, z=z,
                p=2 * stats.norm.sf(abs(z)),
                ci=(diff - 1.96 * se, diff + 1.96 * se), r=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mil", default="runs_mil/predicciones_test_mil.csv")
    ap.add_argument("--cnn", default="runs/predicciones_test.csv")
    args = ap.parse_args()

    mil = pd.read_csv(args.mil)
    cnn = pd.read_csv(args.cnn)
    # El modelo por imagen se agrega promediando las vistas de cada paciente, que
    # es como se reporto en el articulo.
    cnn_bag = cnn.groupby(["patient", "source"], as_index=False).agg(
        prob_cnn=("prob_malignant", "mean"), y_cnn=("y_true", "max"))

    merged = mil.merge(cnn_bag, left_on=["bag", "source"], right_on=["patient", "source"])
    assert (merged.y_true == merged.y_cnn).all(), "las etiquetas de bolsa no coinciden"
    print(f"Bolsas emparejadas: {len(merged)} de {len(mil)} "
          f"(las de cdd-cesm son por mama y no tienen equivalente por paciente)\n")

    print("Prueba de DeLong pareada: MIL frente a modelo por imagen agregado")
    print(f"  {'subconjunto':<14}{'n':>6}{'AUC MIL':>10}{'AUC img':>10}"
          f"{'dif.':>9}{'IC95 de la dif.':>20}{'r':>7}{'p':>10}")
    print("  " + "-" * 78)

    subsets = [("all matched", merged)] + [
        (s, g) for s, g in merged.groupby("source") if g.y_true.nunique() > 1 and len(g) >= 40
    ]
    for name, g in subsets:
        res = compare(g.y_true.values, g.prob_malignant.values, g.prob_cnn.values)
        star = ""
        if not np.isnan(res["p"]):
            star = "  ***" if res["p"] < 0.001 else "  **" if res["p"] < 0.01 \
                else "  *" if res["p"] < 0.05 else "  n.s."
        lo, hi = res["ci"]
        ci_text = f"[{lo:+.3f}, {hi:+.3f}]"
        print(f"  {name:<14}{len(g):>6}{res['auc_a']:>10.3f}{res['auc_b']:>10.3f}"
              f"{res['diff']:>+9.3f}{ci_text:>20}{res['r']:>7.2f}"
              f"{res['p']:>10.4f}{star}")

    print("\n  r es la correlacion entre los dos modelos estimada por DeLong.")
    print("  Cuanto mas alta, mas potencia gana la prueba pareada frente a")
    print("  comparar intervalos de confianza por separado.")


if __name__ == "__main__":
    main()
