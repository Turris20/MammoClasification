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


def _box(ax, x, y, w, h, title, body, accent=False):
    """Caja del diagrama con titulo en negrita y cuerpo debajo."""
    from matplotlib.patches import FancyBboxPatch
    edge = VERMILION if accent else MUTED
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                linewidth=1.1, edgecolor=edge, facecolor="white", zorder=2))
    ax.text(x + w / 2, y + h - 0.052, title, ha="center", va="top", fontsize=7.2,
            fontweight="bold", color=INK, zorder=3)
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top", fontsize=6.6,
            color=MUTED, linespacing=1.5, zorder=3)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color=MUTED,
                                shrinkA=0, shrinkB=0, mutation_scale=9), zorder=1)


def figure_pipeline(outdir):
    """
    Vista general del metodo.

    La fila superior es el tratamiento de los datos, donde estan las dos
    correcciones de protocolo del articulo; la inferior es el modelo y la
    evaluacion. Se destacan en color las tres cajas que constituyen la
    aportacion, para que el lector distinga lo propuesto de lo estandar.
    """
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    # Un margen a ambos lados evita que el trazo de las cajas extremas quede
    # cortado por el recorte ajustado con que se guarda la figura.
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")

    w, h, top, bot = 0.213, 0.30, 0.62, 0.14
    xs = [0.01, 0.265, 0.52, 0.775]

    _box(ax, xs[0], top, w, h, "Multi-source corpus",
         "six public datasets\n12,029 images\n4,044 patients")
    _box(ax, xs[1], top, w, h, "Preprocessing",
         "breast-region crop\nCLAHE, then mask\nbackground set to 0")
    _box(ax, xs[2], top, w, h, "Patient-grouped split",
         "StratifiedGroupKFold\nstratified on\n(source, label)", accent=True)
    _box(ax, xs[3], top, w, h, "Bag construction",
         "level detected\nper dataset:\npatient or breast", accent=True)
    for i in range(3):
        _arrow(ax, xs[i] + w, top + h / 2, xs[i + 1], top + h / 2)

    # Conector de la fila superior a la inferior.
    _arrow(ax, xs[3] + w / 2, top, xs[3] + w / 2, top - 0.09)
    ax.plot([xs[0] + w / 2, xs[3] + w / 2], [top - 0.09, top - 0.09],
            color=MUTED, linewidth=1.0, zorder=1)
    _arrow(ax, xs[0] + w / 2, top - 0.09, xs[0] + w / 2, bot + h)

    _box(ax, xs[0], bot, w, h, "Instance encoder",
         "ConvNeXt-S with\nzero-gated CBAM\non last two stages")
    _box(ax, xs[1], bot, w, h, "Gated attention",
         "weights sum to 1\nwithin each bag;\nviews can be muted")
    _box(ax, xs[2], bot, w, h, "Bag decision",
         "one loss per bag,\nnot per image")
    _box(ax, xs[3], bot, w, h, "Evaluation",
         "within-source AUC\noperating point at\n85% sensitivity", accent=True)
    for i in range(3):
        _arrow(ax, xs[i] + w, bot + h / 2, xs[i + 1], bot + h / 2)

    ax.text(0.5, 0.035, "Coloured outlines mark the components introduced by this work.",
            ha="center", fontsize=6.6, color=VERMILION, style="italic")
    save(fig, outdir, "fig1_pipeline")


def figure_preprocessing(image_path, outdir, clahe_clip=2.0):
    """
    Efecto del preprocesado sobre una imagen real.

    Requiere una mamografia del corpus: el paso que se quiere mostrar es la
    eliminacion del texto quemado y los marcadores, que solo aparecen en las
    imagenes originales y no pueden simularse de forma honesta.
    """
    import cv2
    from cnn_mammo import crop_breast, resize_with_padding

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  (no se pudo leer {image_path}: se omite la Fig. 2)")
        return

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # El umbral por si solo conserva el texto y los marcadores; lo que los
    # elimina es quedarse con la componente conexa mayor. El panel central tiñe
    # la componente retenida para que se vea que se descarta y que se conserva.
    n_cc, cc = cv2.connectedComponents(otsu, connectivity=8)
    keep = 1 + int(np.argmax([(cc == i).sum() for i in range(1, n_cc)])) if n_cc > 1 else 0
    overlay = np.zeros((*otsu.shape, 3), dtype=float)
    overlay[otsu > 0] = [0.72, 0.72, 0.72]
    rgb = tuple(int(VERMILION[i:i + 2], 16) / 255 for i in (1, 3, 5))
    overlay[cc == keep] = rgb

    cropped, mask = crop_breast(gray)
    enhanced = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(cropped)
    final = np.where(mask > 0, np.maximum(enhanced, 1), 0).astype(np.uint8)
    final = resize_with_padding(final, (512, 512))

    panels = [
        (gray, "(a) Raw image", "background, burned-in text\nand annotation artefacts"),
        (overlay, "(b) Otsu threshold", "coloured: largest connected\ncomponent, retained as breast"),
        (final, "(c) Model input", "cropped, CLAHE-enhanced,\nbackground exactly zero"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, (img, title, caption) in zip(axes, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, pad=5)
        ax.text(0.5, -0.06, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=6.6, color=MUTED, linespacing=1.5)
        ax.set_xticks([]); ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_color(GRID)
    fig.tight_layout(w_pad=1.5)
    save(fig, outdir, "fig2_preprocessing")


def pick_contrast_source(d):
    """
    Elige la fuente donde el contraste es limpio.

    Restringir a una sola base es necesario, no cosmetico: las bolsas bilaterales
    no se reparten por igual entre fuentes. cmmd aporta 108 bolsas malignas
    bilaterales y ninguna benigna, porque sus pacientes benignas se fotografiaron
    de un solo lado, asi que un contraste agrupado compara en parte bases de datos
    y no clases. Se toma la fuente que mas bolsas aporta de la clase minoritaria.
    """
    best, best_n = None, 0
    for src, g in d.groupby("source"):
        n = min((g.y == 1).sum(), (g.y == 0).sum())
        if n > best_n:
            best, best_n = src, n
    return best, best_n


def figure_attention(att_path, outdir, source=None):
    """
    Evidencia directa del mecanismo que postula el articulo.

    El cancer es unilateral en el 95-98% de los casos, asi que en una bolsa
    maligna la atencion deberia concentrarse en un lado, mientras que en una
    benigna no hay lesion que encontrar y ningun lado deberia destacar. El panel
    izquierdo contrasta esa prediccion; el derecho muestra bolsas concretas.
    """
    from scipy import stats

    att = pd.read_csv(att_path)
    rows = []
    for (bag, y), g in att.groupby(["bag", "y_true"]):
        sides = g.groupby("laterality").attention.sum()
        if len(sides) >= 2:
            rows.append((bag, y, float(sides.max()), float(g.bag_prob.iloc[0]),
                         g.source.iloc[0]))
    d = pd.DataFrame(rows, columns=["bag", "y", "mass", "prob", "source"])

    if source is None:
        source, n_minor = pick_contrast_source(d)
        print(f"  (atencion: contraste restringido a {source}, "
              f"{n_minor} bolsas en la clase minoritaria)")
    d = d[d.source == source]
    att = att[att.source == source]

    if len(d) < 8:
        print("  (atencion: muy pocas bolsas bilaterales, se omite la figura)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    groups = [("Benign", d[d.y == 0].mass.values, MUTED),
              ("Malignant", d[d.y == 1].mass.values, VERMILION)]
    for i, (label, vals, color) in enumerate(groups):
        jitter = np.random.default_rng(0).normal(0, 0.055, len(vals))
        ax1.scatter(np.full(len(vals), i) + jitter, vals, s=9, color=color,
                    alpha=0.35, edgecolor="none", zorder=2)
        box = ax1.boxplot([vals], positions=[i], widths=0.42, showfliers=False,
                          medianprops=dict(color=INK, linewidth=1.6),
                          boxprops=dict(color=INK, linewidth=1.0),
                          whiskerprops=dict(color=INK, linewidth=1.0),
                          capprops=dict(color=INK, linewidth=1.0))
    ax1.set_xticks([0, 1], [g[0] for g in groups])
    ax1.set_ylabel("Attention mass on dominant laterality")
    ax1.axhline(0.5, color=GRID, linewidth=1.0, zorder=0)
    ax1.text(1.46, 0.505, "evenly split", fontsize=7, color=MUTED,
             ha="right", va="bottom", style="italic")
    ax1.set_xlim(-0.55, 1.55); ax1.set_ylim(0.47, 1.02)

    if len(groups[0][1]) > 1 and len(groups[1][1]) > 1:
        _, pval = stats.mannwhitneyu(groups[1][1], groups[0][1], alternative="greater")
        ax1.set_title(f"{source}  ($n$ = {len(d)})\n"
                      f"Mann–Whitney one-sided  $p$ = {pval:.1e}", fontsize=8, pad=6)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    ax1.grid(axis="y", color=GRID, linewidth=0.5); ax1.set_axisbelow(True)

    # Panel derecho: las dos bolsas malignas mas confiadas y la benigna mas confiada.
    picks = pd.concat([d[d.y == 1].nlargest(2, "prob"), d[d.y == 0].nsmallest(1, "prob")])
    labels, weights, colors = [], [], []
    for _, row in picks.iterrows():
        g = att[att.bag == row.bag].sort_values(["laterality", "view"])
        for _, inst in g.iterrows():
            labels.append(f"{inst.laterality}-{inst.view}")
            weights.append(inst.attention)
            colors.append(VERMILION if row.y == 1 else MUTED)
    ypos = np.arange(len(labels))[::-1]
    ax2.barh(ypos, weights, color=colors, height=0.66)
    ax2.set_yticks(ypos, labels, fontsize=7)
    ax2.set_xlabel("Attention weight $a_k$")
    ax2.set_xlim(0, 1.32)
    # El valor va escrito porque una barra de longitud cero se lee como dato
    # ausente, cuando es justo lo contrario: una vista suprimida por completo,
    # que es el comportamiento que la figura pretende mostrar.
    for yp, w, c in zip(ypos, weights, colors):
        ax2.text(w + 0.02, yp, f"{w:.2f}", va="center", fontsize=6.5,
                 color=c if w > 0.01 else MUTED)

    # Separadores y anotacion por bolsa.
    cursor = 0
    for _, row in picks.iterrows():
        n = len(att[att.bag == row.bag])
        top = ypos[cursor]
        ax2.text(1.30, top,
                 f"{'malignant' if row.y == 1 else 'benign'}\n$p$ = {row.prob:.2f}",
                 fontsize=7, ha="right", va="center",
                 color=VERMILION if row.y == 1 else MUTED)
        cursor += n
        if cursor < len(labels):
            ax2.axhline(ypos[cursor] + 0.5, color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)
    ax2.tick_params(axis="y", length=0)
    ax2.grid(axis="x", color=GRID, linewidth=0.5); ax2.set_axisbelow(True)

    fig.tight_layout(w_pad=2.0)
    save(fig, outdir, "fig5_attention")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mil", default="runs_mil/predicciones_test_mil.csv")
    ap.add_argument("--attention", default="article/attention_test.csv",
                    help="salida de export_attention.py; si no existe se omite la Fig. 5")
    ap.add_argument("--example-image", default=None,
                    help="ruta a una mamografia del corpus para la Fig. 2; conviene "
                         "elegir una con texto quemado o marcadores visibles")
    ap.add_argument("--attention-source", default=None,
                    help="fuente a la que restringir la Fig. 5; por defecto se elige "
                         "la que mas bolsas aporta de la clase minoritaria")
    ap.add_argument("--outdir", default="article/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup_style()
    df = pd.read_csv(args.mil)
    print(f"{len(df)} bolsas, AUC {roc_auc_score(df.y_true, df.prob_malignant):.4f}")
    print("Figuras escritas:")
    figure_pipeline(args.outdir)             # Fig. 1
    if args.example_image:
        figure_preprocessing(args.example_image, args.outdir)   # Fig. 2
    else:
        print("  (sin --example-image: se omite la Fig. 2)")
    figure_pair_weights(df, args.outdir)      # Fig. 3
    figure_label_noise(args.outdir)          # Fig. 4
    if os.path.exists(args.attention):
        figure_attention(args.attention, args.outdir, args.attention_source)   # Fig. 5
    else:
        print(f"  (sin {args.attention}: se omite la Fig. 5)")
    figure_roc(df, args.outdir)              # Fig. 6
    figure_confusion(df, args.outdir)        # Fig. 7


if __name__ == "__main__":
    main()
