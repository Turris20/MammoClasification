"""
Mide la resolucion nativa de las imagenes y la resolucion efectiva que queda
despues de redimensionar.

Para que sirve
--------------
En CMMD el 44% de los casos malignos involucran calcificaciones, y una
microcalcificacion mide entre 0.2 y 0.5 mm. Con un detector FFDM tipico
(70 um/pixel), redimensionar a 512 deja cada pixel en 0.3-0.55 mm: la
calcificacion cae por debajo de un pixel y desaparece. Solo sobreviven las
masas (5-30 mm).

Antes de invertir horas entrenando a mayor resolucion conviene comprobar si las
imagenes de origen tienen detalle que recuperar: si los JPEG ya vienen
downsampleados a ~1000 px, subir el tamano de entrada no aporta nada.

Uso:
    python check_resolution.py --data-dir splits
    python check_resolution.py --data-dir splits --sample 200
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# Paso de pixel tipico de un detector de mamografia digital de campo completo.
DETECTOR_UM_PER_PX = 70.0
# Rango de tamano de una microcalcificacion.
MICROCALC_MM = (0.2, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="splits")
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--sample", type=int, default=120, help="imagenes a medir por dataset")
    ap.add_argument("--targets", type=int, nargs="+", default=[512, 768, 1024, 1536])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = []
    for name in ["train", "val", "test"]:
        path = os.path.join(args.data_dir, f"{name}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit(f"No se encontraron CSV en {args.data_dir}")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]

    rng = np.random.default_rng(args.seed)
    dims = defaultdict(list)

    for source, group in df.groupby("source_dataset"):
        idx = rng.permutation(len(group))[: args.sample]
        for i in idx:
            rel = str(group.iloc[i]["raw_image_path"])
            path = rel if os.path.isabs(rel) else os.path.join(args.base_dir, rel)
            if not os.path.exists(path):
                continue
            try:
                with Image.open(path) as im:  # solo lee la cabecera, no decodifica
                    dims[source].append(im.size)  # (ancho, alto)
            except Exception:
                continue

    if not dims:
        raise SystemExit("No se pudo abrir ninguna imagen. Revisa --base-dir.")

    print(f"{'dataset':<16}{'n':>5}{'ancho med.':>12}{'alto med.':>11}{'lado mayor':>12}")
    print("-" * 56)
    longest = {}
    for source in sorted(dims):
        arr = np.array(dims[source])
        w, h = int(np.median(arr[:, 0])), int(np.median(arr[:, 1]))
        longest[source] = max(w, h)
        print(f"{source:<16}{len(arr):>5}{w:>12}{h:>11}{longest[source]:>12}")

    print(f"\nResolucion efectiva en mm/pixel (detector a {DETECTOR_UM_PER_PX:.0f} um/px)")
    print("Una microcalcificacion mide "
          f"{MICROCALC_MM[0]}-{MICROCALC_MM[1]} mm; por encima de "
          f"{MICROCALC_MM[0]} mm/px ya es sub-pixel.\n")
    header = f"  {'dataset':<16}" + "".join(f"{t:>10}" for t in args.targets)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for source in sorted(longest):
        row = f"  {source:<16}"
        for target in args.targets:
            # Redimensionar conserva la relacion de aspecto, asi que el factor lo
            # marca el lado mayor.
            mm = (longest[source] * DETECTOR_UM_PER_PX / 1000.0) / target
            row += f"{mm:>9.2f}{'*' if mm <= MICROCALC_MM[0] else ' '}"
        print(row)
    print("\n  (*) resolucion suficiente para que una microcalcificacion ocupe >= 1 pixel")

    print("\nLectura:")
    print("  - Si el lado mayor nativo ya es <= al tamano de entrada, subir la")
    print("    resolucion solo interpola y no aporta informacion nueva.")
    print("  - Si el nativo es mucho mayor, hay detalle real que se esta tirando y")
    print("    entrenar a mayor resolucion si puede mover las metricas.")


if __name__ == "__main__":
    main()
