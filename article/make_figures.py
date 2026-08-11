"""
Genera las figuras del articulo a partir de las predicciones guardadas.

    python article/make_figures.py --mil runs_mil/predicciones_test_mil.csv \
        --outdir article/figures

Salida en PDF (vectorial, para someter) y PNG (para revisar). La paleta es
Okabe-Ito, validada para deficiencias de vision del color, y cada serie lleva
ademas un estilo de linea distinto para que las figuras sobrevivan a la
impresion en escala de grises.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Paleta Okabe-Ito. Validada: separacion CVD minima ΔE 11.0 (deuteranopia),
# 25.8 en vision normal, contraste >= 3:1 sobre fondo claro.
BLUE, VERMILION, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, MUTED, GRID = "#1a1a1a", "#666666", "#d9d9d9"

# Umbral fijado en validacion con el criterio de sensibilidad >= 85%.
THRESHOLD = 0.218
# Puntos de operacion medidos en validacion (del log de entrenamiento). El
# desplazamiento de la etiqueta se fija por punto porque Youden y el criterio de
# especificidad caen muy juntos sobre la curva de test y las anotaciones chocan.
OPERATING_POINTS = [
    ("Youden", 0.800, (58, -54)),
    (r"sens $\geq$ 85%", 0.218, (14, -22)),
    (r"spec $\geq$ 90%", 0.677, (74, -20)),
]


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """Intervalo de confianza de Hanley-McNeil, el mismo que usa el pipeline."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(0.0, var))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def save(fig, outdir, name):
    for ext in ("pdf", "png"):
        path = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(path)
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def figure_roc(df, outdir):
    """
    ROC global y de las dos fuentes que concentran el 96% de la evidencia.

    Se marcan los tres puntos de operacion candidatos sobre la curva global para
    que el compromiso sensibilidad-especificidad sea legible, en vez de dejar el
    AUC como un numero sin consecuencia clinica.
    """
    fig, ax = plt.subplots(figsize=(4.4, 4.0))

    series = [
        ("All sources", df, BLUE, "-", 2.0),
        ("cmmd", df[df.source == "cmmd"], VERMILION, "--", 1.5),
        ("mini-ddsm", df[df.source == "mini-ddsm"], GREEN, "-.", 1.5),
    ]

    for label, sub, color, style, width in series:
        y, p = sub.y_true.values, sub.prob_malignant.values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        lo, hi = auc_ci(auc, int(y.sum()), int((1 - y).sum()))
        ax.plot(fpr, tpr, color=color, linestyle=style, linewidth=width,
                label=f"{label}  AUC {auc:.3f} [{lo:.3f}, {hi:.3f}]")

    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.0, zorder=0)
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=7, rotation=38, style="italic")

    # Puntos de operacion sobre la curva global.
    y, p = df.y_true.values, df.prob_malignant.values
    for name, thr, offset in OPERATING_POINTS:
        pred = (p >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum(); tn = ((pred == 0) & (y == 0)).sum()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        chosen = abs(thr - THRESHOLD) < 1e-9
        ax.plot(1 - spec, sens, marker="o", markersize=7 if chosen else 5.5,
                markerfacecolor=INK if chosen else "white",
                markeredgecolor=INK, markeredgewidth=1.2, zorder=5)
        ax.annotate(f"{name}\n({(1-spec)*100:.0f}%, {sens*100:.0f}%)",
                    xy=(1 - spec, sens), xytext=offset, textcoords="offset points",
                    fontsize=7, color=INK if chosen else MUTED,
                    fontweight="bold" if chosen else "normal",
                    arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                    color=INK if chosen else MUTED,
                                    shrinkA=1, shrinkB=4))

    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False, handlelength=2.4)
    save(fig, outdir, "fig6_roc")


def figure_confusion(df, outdir):
    """Matriz de confusion en el punto de operacion elegido en validacion."""
    y = df.y_true.values
    pred = (df.prob_malignant.values >= THRESHOLD).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    # Rampa secuencial de un solo tono: la magnitud es una cantidad, no categorias.
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)

    names = ["Benign", "Malignant"]
    ax.set_xticks([0, 1], names); ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Reference")

    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n{frac * 100:.1f}%", ha="center", va="center",
                    fontsize=9, color="white" if frac > 0.5 else INK)

    tn, fp, fn, tp = cm.ravel()
    ax.set_title(f"Sensitivity {tp / (tp + fn) * 100:.1f}%   "
                 f"Specificity {tn / (tn + fp) * 100:.1f}%", pad=8, fontsize=8)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    save(fig, outdir, "fig7_confusion")


def figure_label_noise(outdir):
    """
    La prediccion central del articulo: el salto de AUC al agregar por paciente
    debe crecer con la fraccion contaminada f. Es la evidencia de que el techo
    lo imponia la anotacion y no el modelo, asi que merece una figura y no solo
    una fila de tabla.
    """
    data = [("mini-ddsm", 0.50, 0.647, 0.761), ("cmmd", 0.36, 0.676, 0.717),
            ("cdd-cesm", 0.02, 0.806, 0.803)]
    f = np.array([d[1] for d in data])
    gain = np.array([d[3] - d[2] for d in data])

    fig, ax = plt.subplots(figsize=(4.6, 3.5))

    # Eq. (6) da AUC_img = (1-f)*AUC_true + 0.5f y, si agregar elimina la
    # contaminacion, AUC_pat = AUC_true. La ganancia es entonces f*(AUC_true - 0.5):
    # una recta por el origen. Se ajusta la pendiente por minimos cuadrados en vez
    # de imponer un AUC_true, porque la agregacion no elimina el ruido por completo
    # -- el modelo se entreno con las etiquetas contaminadas -- y fijarlo a priori
    # sobreestimaria la magnitud del salto.
    slope = float((f * gain).sum() / (f ** 2).sum())
    grid = np.linspace(0, 0.55, 50)
    ax.plot(grid, slope * grid, color=MUTED, linestyle=":", linewidth=1.3, zorder=1,
            label=r"Eq. (6) fit: $f\,(\mathrm{AUC}_{\mathrm{true}} - 0.5)$,"
                  f"  slope {slope:.3f}")

    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.scatter(f, gain, s=70, color=BLUE, zorder=3, edgecolor="white", linewidth=1.2)
    for name, fi, gi in zip([d[0] for d in data], f, gain):
        ax.annotate(name, xy=(fi, gi), xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)

    r = np.corrcoef(f, gain)[0, 1]
    ax.text(0.03, 0.95,
            f"$r$ = {r:.2f}\nimplied AUC$_{{\mathrm{{true}}}}$ = {0.5 + slope:.2f}",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)

    ax.set_xlabel(r"Contaminated fraction of positives, $f$")
    ax.set_ylabel("AUC gain from patient-level aggregation")
    ax.set_xlim(-0.02, 0.56)
    # Margen superior para que la etiqueta de la serie mas alta no toque el borde.
    ax.set_ylim(gain.min() - 0.015, gain.max() + 0.030)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower right", frameon=False)
    save(fig, outdir, "fig4_label_noise")


def figure_pair_weights(df, outdir):
    """
    Cuanta evidencia aporta realmente cada fuente al AUC intra-fuente.

    El numero de bolsas engaña: el peso es el producto de las clases, asi que
    dmid y kau-bcmd desaparecen pese a mostrar AUC de 1.000.
    """
    rows = []
    for s, g in df.groupby("source"):
        npos, nneg = int(g.y_true.sum()), int((1 - g.y_true).sum())
        if npos * nneg == 0:
            continue
        rows.append((s, npos * nneg, roc_auc_score(g.y_true, g.prob_malignant)))
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    pairs = np.array([r[1] for r in rows], dtype=float)
    aucs = [r[2] for r in rows]
    share = pairs / pairs.sum() * 100

    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    colors = [BLUE if s >= 5 else MUTED for s in share]
    bars = ax.barh(names, share, color=colors, height=0.62)
    for bar, s, a in zip(bars, share, aucs):
        ax.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}%   AUC {a:.3f}", va="center", fontsize=8,
                color=INK if s >= 5 else MUTED)

    ax.set_xlabel("Share of within-source benign–malignant pairs (%)")
    ax.set_xlim(0, 78)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, outdir, "fig3_pair_weights")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mil", default="runs_mil/predicciones_test_mil.csv")
    ap.add_argument("--outdir", default="article/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()
    df = pd.read_csv(args.mil)
    print(f"{len(df)} bolsas, AUC {roc_auc_score(df.y_true, df.prob_malignant):.4f}")
    print("Figuras escritas:")
    figure_roc(df, args.outdir)
    figure_confusion(df, args.outdir)
    figure_label_noise(args.outdir)
    figure_pair_weights(df, args.outdir)


if __name__ == "__main__":
    main()
