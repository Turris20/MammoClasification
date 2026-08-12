"""
Clasificador de mamografia con Multiple Instance Learning y atencion.

Por que
-------
En cmmd, mini-ddsm y kau-bcmd el 100% de los pacientes con ambas mamas
fotografiadas comparten etiqueta en las dos: el diagnostico es por paciente y se
propago a cada imagen. Como el cancer de mama es unilateral en el 95-98% de los
casos, alrededor de la mitad de las imagenes etiquetadas "maligno" muestran una
mama sana (f=0.36 en cmmd, f=0.50 en mini-ddsm).

Entrenando por imagen, la red mira esa mama sana, dice correctamente "benigno" y
recibe un gradiente que la castiga por acertar. MIL reformula el problema: el
paciente es una bolsa de imagenes y la bolsa es positiva si al menos una
instancia lo es. La perdida se calcula una sola vez por bolsa, de modo que la red
queda libre de puntuar bajo la mama sana mientras puntue alto la afectada.

El pooling por atencion (Ilse et al., 2018) ademas expone que imagen dirigio la
decision, o sea localizacion aproximada sin necesitar mascaras ni ROI.

Uso:
    python mil_mammo.py --data-dir splits --cache-dir cache1024 \
        --arch convnext_small --img-size 1024 --bag-batch 1 --accum 16
"""

import argparse
import math
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from cnn_mammo import (
    BACKBONES,
    ConvNeXtCBAM,
    MammogramDataset,
    ModelEMA,
    auc_ci,
    best_threshold,
    build_param_groups,
    print_operating_points,
    cosine_with_warmup,
    plot_confusion,
    plot_history,
)


# ---------------------------------------------------------------------------
# Construccion de bolsas
# ---------------------------------------------------------------------------

def detect_bag_levels(df, threshold=0.9):
    """
    Decide, por base de datos, si la bolsa debe ser el paciente o la mama.

    Si dentro de un dataset las dos mamas de un paciente casi siempre comparten
    etiqueta, la anotacion es de paciente y agrupar por paciente es lo correcto.
    Si la etiqueta cambia entre lados (cdd-cesm: solo el 37% coinciden), la
    anotacion ya es por mama y agrupar por paciente destruiria esa granularidad.
    """
    levels = {}
    for source, group in df.groupby("source_dataset"):
        same = []
        for _, g in group.groupby("patient_key"):
            lat = set(g["laterality"].dropna())
            if {"L", "R"} <= lat:
                same.append(g[g.laterality == "L"].label.max() == g[g.laterality == "R"].label.max())
        levels[source] = "patient" if (not same or np.mean(same) >= threshold) else "breast"
    return levels


def build_bags(dataset, bag_levels):
    """
    Agrupa los indices del dataset en bolsas y devuelve etiqueta y fuente de cada una.

    La etiqueta de la bolsa es el maximo de sus instancias: positiva si alguna lo es.
    """
    df = dataset.df
    keys = []
    for i in range(len(df)):
        row = df.iloc[i]
        source = str(row["source_dataset"])
        key = dataset.patients[i]
        if bag_levels.get(source, "patient") == "breast":
            key = f"{key}|{row.get('laterality', '?')}"
        keys.append(key)

    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)

    bag_keys = list(groups)
    indices = [groups[k] for k in bag_keys]
    labels = np.array([int(max(dataset.labels[j] for j in g)) for g in indices])
    sources = np.array([dataset.sources[g[0]] for g in indices])
    return bag_keys, indices, labels, sources


class BagDataset(torch.utils.data.Dataset):
    """Envuelve un MammogramDataset por imagen y devuelve bolsas."""

    def __init__(self, image_ds, bag_levels, max_bag=0, train=False):
        self.ds = image_ds
        self.keys, self.indices, self.labels, self.sources = build_bags(image_ds, bag_levels)
        self.max_bag = max_bag
        self.train = train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, b):
        idx = self.indices[b]
        # Recortar solo en entrenamiento y al azar: en evaluacion interesa usar
        # todas las vistas disponibles del paciente.
        if self.train and self.max_bag and len(idx) > self.max_bag:
            idx = list(np.random.choice(idx, self.max_bag, replace=False))
        images = torch.stack([self.ds[i][0] for i in idx])
        return images, torch.tensor(int(self.labels[b]), dtype=torch.long)


def collate_bags(batch):
    """
    Aplana las bolsas en un solo tensor mas un vector que dice a que bolsa
    pertenece cada imagen. Evita rellenar bolsas de tamano distinto.
    """
    images = torch.cat([b[0] for b in batch], dim=0)
    bag_idx = torch.cat([torch.full((len(b[0]),), i, dtype=torch.long)
                         for i, b in enumerate(batch)])
    labels = torch.stack([b[1] for b in batch])
    return images, bag_idx, labels


def make_bag_sampler(bag_ds):
    """Iguala el peso de cada celda (dataset, clase) contando bolsas, no imagenes."""
    cells = [f"{s}|{l}" for s, l in zip(bag_ds.sources, bag_ds.labels)]
    counts = defaultdict(int)
    for c in cells:
        counts[c] += 1
    weights = torch.tensor([1.0 / counts[c] for c in cells], dtype=torch.double)
    return torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def segment_softmax(scores, bag_idx, n_bags):
    """Softmax aplicado dentro de cada bolsa por separado."""
    m = torch.full((n_bags,), float("-inf"), device=scores.device, dtype=scores.dtype)
    m = m.scatter_reduce(0, bag_idx, scores, reduce="amax", include_self=True)
    e = torch.exp(scores - m[bag_idx])
    s = torch.zeros(n_bags, device=scores.device, dtype=scores.dtype).scatter_add(0, bag_idx, e)
    return e / (s[bag_idx] + 1e-8)


class GatedAttentionMIL(nn.Module):
    """
    Atencion con compuerta de Ilse et al. (2018).

        a_k = softmax( w^T ( tanh(V h_k) * sigmoid(U h_k) ) )
        z   = sum_k a_k h_k

    La rama sigmoide deja que la red suprima instancias irrelevantes -- que es
    exactamente el papel de la mama contralateral sana -- en vez de promediarlas
    con el mismo peso que la afectada.
    """

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, h, bag_idx, n_bags):
        scores = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = segment_softmax(scores.float(), bag_idx, n_bags).to(h.dtype)
        z = torch.zeros(n_bags, h.shape[1], device=h.device, dtype=h.dtype)
        z = z.index_add(0, bag_idx, a.unsqueeze(-1) * h)
        return z, a


class MILNet(nn.Module):
    def __init__(self, arch="convnext_small", pretrained=True, cbam_stages=2,
                 dropout=0.5, drop_path=0.0, freeze_stages=0, attn_hidden=256):
        super().__init__()
        base = ConvNeXtCBAM(arch, 2, pretrained, cbam_stages, dropout,
                            drop_path=drop_path, freeze_stages=freeze_stages)
        self.backbone = base.backbone
        self.cbam = base.cbam
        # La cabeza de clasificacion original se sustituye: aqui el encoder solo
        # produce el embedding de cada imagen y la decision se toma tras agrupar.
        dim = next((m.in_features for m in base.backbone.classifier
                    if isinstance(m, nn.Linear)), None)
        if dim is None:
            raise RuntimeError("no se pudo inferir la dimension del embedding")
        self.embed_norm = nn.LayerNorm(dim)
        self.attention = GatedAttentionMIL(dim, attn_hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, 2))
        self.dim = dim

    def encode(self, images):
        x = images
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            key = str(i)
            if key in self.cbam:
                x = self.cbam[key](x)
        x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.embed_norm(x)

    def forward(self, images, bag_idx, n_bags, return_attention=False):
        h = self.encode(images)
        z, a = self.attention(h, bag_idx, n_bags)
        logits = self.head(z)
        return (logits, a) if return_attention else logits


# ---------------------------------------------------------------------------
# Entrenamiento y evaluacion
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler,
                    ema, accum=1, clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, bag_idx, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        bag_idx = bag_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images, bag_idx, labels.size(0))
            loss = criterion(logits, labels)

        scaler.scale(loss / accum).backward()

        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / max(1, total), correct / max(1, total)


@torch.no_grad()
def infer(model, loader, criterion, device, tta=False, chunk=8):
    model.eval()
    total_loss, total = 0.0, 0
    probs, targets = [], []

    for images, bag_idx, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        bag_idx = bag_idx.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            # Sin gradiente se pueden codificar las imagenes por trozos, asi que
            # una bolsa grande no obliga a tener todas sus vistas en memoria a la vez.
            h = torch.cat([model.encode(images[i:i + chunk])
                           for i in range(0, len(images), chunk)])
            z, _ = model.attention(h, bag_idx, labels.size(0))
            logits = model.head(z)
            loss = criterion(logits, labels)
            p = torch.softmax(logits.float(), 1)[:, 1]
            if tta:
                hf = torch.cat([model.encode(torch.flip(images[i:i + chunk], dims=[3]))
                                for i in range(0, len(images), chunk)])
                zf, _ = model.attention(hf, bag_idx, labels.size(0))
                p = (p + torch.softmax(model.head(zf).float(), 1)[:, 1]) / 2

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        probs.append(p.cpu())
        targets.append(labels.cpu())

    return total_loss / max(1, total), torch.cat(probs).numpy(), torch.cat(targets).numpy()


def report(y_true, probs, threshold, sources, title="TEST"):
    preds = (probs >= threshold).astype(int)
    npos, nneg = int((y_true == 1).sum()), int((y_true == 0).sum())
    auc = roc_auc_score(y_true, probs)
    lo, hi = auc_ci(auc, npos, nneg)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    print(f"\n{'=' * 62}\n{title} - por bolsa ({len(y_true)})  umbral {threshold:.3f}\n{'=' * 62}")
    print(f"AUC-ROC:       {auc:.4f}  IC95 [{lo:.3f}, {hi:.3f}]")
    print(f"Accuracy:      {accuracy_score(y_true, preds) * 100:.2f}%")
    print(f"Recall macro:  {recall_score(y_true, preds, average='macro') * 100:.2f}%")
    print(f"F1 macro:      {f1_score(y_true, preds, average='macro'):.4f}")
    print(f"Sensibilidad:  {tp / max(1, npos) * 100:.2f}%")
    print(f"Especificidad: {tn / max(1, nneg) * 100:.2f}%")

    print(f"\n  {'dataset':<14}{'bolsas':>8}{'pares':>8}{'AUC':>7}{'IC 95%':>16}")
    wp, wa = 0.0, 0.0
    for source in sorted(set(sources)):
        m = sources == source
        yt, pr = y_true[m], probs[m]
        a, b = int((yt == 1).sum()), int((yt == 0).sum())
        if a * b == 0:
            continue
        s_auc = roc_auc_score(yt, pr)
        l, h = auc_ci(s_auc, a, b)
        wp += a * b
        wa += a * b * s_auc
        flag = "  <-- pocos datos" if a * b < 500 else ""
        print(f"  {source:<14}{int(m.sum()):>8}{a * b:>8}{s_auc:>7.3f}{f'[{l:.3f}, {h:.3f}]':>16}{flag}")
    if wp:
        print(f"\n  AUC DENTRO de fuente:  {wa / wp:.4f}   <-- la cifra a reportar")
    return auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="splits")
    ap.add_argument("--base-dir", default="")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--outdir", default="runs_mil")
    ap.add_argument("--arch", default="convnext_small", choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=1024)
    ap.add_argument("--bag-batch", type=int, default=1, help="bolsas por paso")
    ap.add_argument("--max-bag", type=int, default=4, help="tope de imagenes por bolsa en train")
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--cbam-stages", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--drop-path", type=float, default=0.4)
    ap.add_argument("--freeze-stages", type=int, default=0)
    ap.add_argument("--attn-hidden", type=int, default=256)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bag-level", choices=["auto", "patient", "breast"], default="auto")
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--no-bg-mask", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--target-sens", type=float, default=0.85,
                    help="sensibilidad minima al fijar el umbral; 0 usa Youden")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.target_sens <= 0:
        args.target_sens = None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    img_size = (args.img_size, args.img_size)
    ds_kwargs = dict(base_dir=args.base_dir, img_size=img_size, cache_dir=args.cache_dir,
                     do_crop=not args.no_crop, bg_mask=not args.no_bg_mask)
    image_ds = {name: MammogramDataset(os.path.join(args.data_dir, f"{name}.csv"),
                                       "train" if name == "train" else name, **ds_kwargs)
                for name in ["train", "val", "test"]}

    if args.bag_level == "auto":
        bag_levels = detect_bag_levels(image_ds["train"].df.assign(
            patient_key=image_ds["train"].patients))
        print("Nivel de bolsa detectado por dataset:")
        for k, v in sorted(bag_levels.items()):
            print(f"  {k:<14} -> {v}")
    else:
        bag_levels = defaultdict(lambda: args.bag_level)

    bags = {name: BagDataset(ds, bag_levels, args.max_bag, train=(name == "train"))
            for name, ds in image_ds.items()}
    for name, b in bags.items():
        sizes = [len(i) for i in b.indices]
        print(f"{name}: {len(b)} bolsas de {len(b.ds)} imagenes "
              f"(mediana {int(np.median(sizes))}, max {max(sizes)}), "
              f"{b.labels.mean() * 100:.1f}% positivas")

    dl = dict(num_workers=args.workers, pin_memory=True, collate_fn=collate_bags,
              persistent_workers=args.workers > 0)
    train_loader = torch.utils.data.DataLoader(
        bags["train"], batch_size=args.bag_batch, sampler=make_bag_sampler(bags["train"]), **dl)
    val_loader = torch.utils.data.DataLoader(bags["val"], batch_size=args.bag_batch,
                                             shuffle=False, **dl)
    test_loader = torch.utils.data.DataLoader(bags["test"], batch_size=args.bag_batch,
                                              shuffle=False, **dl)

    model = MILNet(args.arch, True, args.cbam_stages, args.dropout,
                   args.drop_path, args.freeze_stages, args.attn_hidden).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(build_param_groups(model, args.lr, args.weight_decay, args.layer_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    steps = max(1, math.ceil(len(train_loader) / args.accum))
    scheduler = cosine_with_warmup(optimizer, steps, steps * args.epochs)
    ema = None if args.no_ema else ModelEMA(model)

    history = defaultdict(list)
    best_auc, best_epoch = -1.0, -1
    ckpt = os.path.join(args.outdir, f"best_mil_{args.arch}.pth")

    for epoch in range(0 if args.eval_only else args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          scheduler, device, scaler, ema, args.accum)

        def evaluate(m):
            loss, p, y = infer(m, val_loader, criterion, device)
            a = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
            return loss, a, accuracy_score(y, (p >= 0.5).astype(int))

        cands = [("raw", model) + evaluate(model)]
        if ema is not None:
            cands.append(("ema", ema.module) + evaluate(ema.module))
        tag, best_m, va_loss, va_auc, va_acc = max(cands, key=lambda c: c[3])

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        lr_now = max(g["lr"] for g in optimizer.param_groups)
        print(f"Epoca {epoch + 1}/{args.epochs}  ({time.time() - t0:.0f}s, lr={lr_now:.2e})")
        print(f"  train  loss {tr_loss:.4f} | acc {tr_acc * 100:.2f}%")
        for n, _, l, a, ac in cands:
            print(f"  val[{n}]  loss {l:.4f} | acc {ac * 100:.2f}% | AUC {a:.4f}"
                  f"{' <-' if n == tag else ''}")

        if va_auc > best_auc:
            best_auc, best_epoch = va_auc, epoch
            torch.save({"epoch": epoch, "model_state_dict": best_m.state_dict(),
                        "val_auc": float(va_auc), "variant": tag,
                        "bag_levels": dict(bag_levels), "args": vars(args)}, ckpt)
            print(f"  -> mejor modelo guardado ({tag}, AUC {va_auc:.4f})")

    if history:
        plot_history(history, args.outdir)

    if not os.path.exists(ckpt):
        raise SystemExit(f"No existe el checkpoint {ckpt}")
    try:
        state = torch.load(ckpt, map_location=device)
    except Exception:
        state = torch.load(ckpt, map_location=device, weights_only=False)
    print(f"\nCargando checkpoint: epoca {state.get('epoch', -1) + 1}, "
          f"AUC val {float(state.get('val_auc', float('nan'))):.4f}, "
          f"variante {state.get('variant', '?')}")
    model.load_state_dict(state["model_state_dict"])

    _, va_p, va_y = infer(model, val_loader, criterion, device, tta=True)
    print_operating_points(va_y, va_p)
    thr = best_threshold(va_y, va_p, target_sens=args.target_sens)
    report(va_y, va_p, thr, bags["val"].sources, "VALIDACION")

    _, te_p, te_y = infer(model, test_loader, criterion, device, tta=True)
    report(te_y, te_p, thr, bags["test"].sources, "TEST")
    plot_confusion(te_y, (te_p >= thr).astype(int), args.outdir)

    pd.DataFrame({"bag": bags["test"].keys, "source": bags["test"].sources,
                  "y_true": te_y, "prob_malignant": te_p}).to_csv(
        os.path.join(args.outdir, "predicciones_test_mil.csv"), index=False)
    print(f"\nSalidas escritas en {args.outdir}/")


if __name__ == "__main__":
    main()
