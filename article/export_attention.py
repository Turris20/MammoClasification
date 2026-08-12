"""
Exporta los pesos de atencion del modelo MIL, instancia por instancia.

Para que
--------
El articulo afirma que la atencion con compuerta permite suprimir la mama
contralateral sana. Esa afirmacion es comprobable sin anotacion de lesion.

El cancer de mama es unilateral en el 95-98% de los casos. Si el mecanismo opera
como se postula, en una bolsa maligna la atencion deberia concentrarse en una
lateralidad, mientras que en una bolsa benigna no hay lesion que encontrar y
ningun lado deberia destacar. El estadistico

    m = max( suma de a_k sobre L , suma de a_k sobre R )

va de 0.5 (repartida por igual entre los dos lados) a 1.0 (toda en un lado), y
la prediccion es m(malignas) > m(benignas). Si no se cumple, el mecanismo no
esta operando como se afirma y el articulo debe decirlo.

Uso:
    python article/export_attention.py --data-dir splits --cache-dir cache1024 \
        --ckpt runs_mil/best_mil_convnext_small.pth --arch convnext_small \
        --img-size 1024 --out article/attention_test.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from cnn_mammo import BACKBONES, MammogramDataset
from mil_mammo import BagDataset, MILNet, collate_bags, detect_bag_levels


@torch.no_grad()
def export(model, bag_ds, loader, device, chunk=8):
    """Devuelve una fila por instancia con su peso de atencion dentro de la bolsa."""
    model.eval()
    df = bag_ds.ds.df
    rows = []
    bag_cursor = 0

    for images, bag_idx, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        bag_idx = bag_idx.to(device, non_blocking=True)
        n_bags = labels.size(0)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            h = torch.cat([model.encode(images[i:i + chunk])
                           for i in range(0, len(images), chunk)])
            z, attn = model.attention(h, bag_idx, n_bags)
            prob = torch.softmax(model.head(z).float(), 1)[:, 1]

        attn = attn.float().cpu().numpy()
        prob = prob.cpu().numpy()
        bag_idx_np = bag_idx.cpu().numpy()

        for b in range(n_bags):
            gi = bag_cursor + b
            instances = bag_ds.indices[gi]
            weights = attn[bag_idx_np == b]
            # El collate concatena las bolsas en orden, asi que el orden de los
            # pesos coincide con el de los indices de la bolsa.
            assert len(weights) == len(instances), "desalineacion entre pesos e instancias"
            for w, idx in zip(weights, instances):
                row = df.iloc[idx]
                rows.append(dict(
                    bag=bag_ds.keys[gi],
                    source=bag_ds.sources[gi],
                    y_true=int(bag_ds.labels[gi]),
                    bag_prob=float(prob[b]),
                    bag_size=len(instances),
                    laterality=row.get("laterality", "?"),
                    view=row.get("view", "?"),
                    attention=float(w),
                    image_path=row["abs_path"],
                ))
        bag_cursor += n_bags

    return pd.DataFrame(rows)


def summarize(att):
    """Contrasta la prediccion de concentracion lateral en bolsas malignas."""
    from scipy import stats

    both = []
    for (bag, y), g in att.groupby(["bag", "y_true"]):
        sides = g.groupby("laterality").attention.sum()
        if len(sides) < 2:
            continue  # una sola lateralidad: el estadistico seria 1.0 por definicion
        both.append(dict(bag=bag, y_true=y, dominant_mass=float(sides.max()),
                         n_views=len(g)))
    d = pd.DataFrame(both)
    if d.empty:
        print("\nNo hay bolsas con las dos lateralidades; el contraste no aplica.")
        return d

    mal = d[d.y_true == 1].dominant_mass
    ben = d[d.y_true == 0].dominant_mass
    print(f"\nBolsas con ambas lateralidades: {len(d)} "
          f"({len(mal)} malignas, {len(ben)} benignas)")
    print(f"  masa de atencion en la lateralidad dominante")
    print(f"    malignas: media {mal.mean():.3f}  mediana {mal.median():.3f}")
    print(f"    benignas: media {ben.mean():.3f}  mediana {ben.median():.3f}")
    if len(mal) > 1 and len(ben) > 1:
        u, p = stats.mannwhitneyu(mal, ben, alternative="greater")
        print(f"  Mann-Whitney unilateral (malignas > benignas): U={u:.0f}, p={p:.4f}")
        print("  " + ("La prediccion se cumple." if p < 0.05 else
                      "La prediccion NO se cumple: reportarlo como tal."))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="splits")
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--ckpt", default="runs_mil/best_mil_convnext_small.pth")
    ap.add_argument("--arch", default="convnext_small", choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--bag-batch", type=int, default=1)
    ap.add_argument("--cbam-stages", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-bg-mask", action="store_true")
    ap.add_argument("--out", default="article/attention_test.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    ds_kwargs = dict(base_dir=args.base_dir, img_size=(args.img_size, args.img_size),
                     cache_dir=args.cache_dir, do_crop=not args.no_crop,
                     bg_mask=not args.no_bg_mask)
    train_ds = MammogramDataset(os.path.join(args.data_dir, "train.csv"), "val", **ds_kwargs)
    bag_levels = detect_bag_levels(train_ds.df.assign(patient_key=train_ds.patients))

    image_ds = MammogramDataset(os.path.join(args.data_dir, f"{args.split}.csv"),
                                args.split, **ds_kwargs)
    bags = BagDataset(image_ds, bag_levels, 0, train=False)
    loader = torch.utils.data.DataLoader(bags, batch_size=args.bag_batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True,
                                         collate_fn=collate_bags)

    model = MILNet(args.arch, False, args.cbam_stages, args.dropout).to(device)
    model = model.to(memory_format=torch.channels_last)
    try:
        state = torch.load(args.ckpt, map_location=device)
    except Exception:
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"Checkpoint: epoca {state.get('epoch', -1) + 1}, "
          f"AUC val {float(state.get('val_auc', float('nan'))):.4f}")

    att = export(model, bags, loader, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    att.to_csv(args.out, index=False)
    print(f"\n{len(att)} instancias de {att.bag.nunique()} bolsas -> {args.out}")

    # Comprobacion de integridad: la atencion debe sumar 1 dentro de cada bolsa.
    sums = att.groupby("bag").attention.sum()
    assert np.allclose(sums, 1.0, atol=1e-4), "la atencion no suma 1 en alguna bolsa"
    print("Verificado: la atencion suma 1.000 en todas las bolsas")

    summarize(att)


if __name__ == "__main__":
    main()
