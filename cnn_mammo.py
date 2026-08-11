"""
Clasificador benigno/maligno de mamografia digital sobre multiples bases de datos.

Backbone ConvNeXt + CBAM con compuerta residual, entrenamiento con warmup+coseno,
EMA de pesos, muestreo balanceado por (dataset, clase) y evaluacion desglosada
por fuente y agregada por paciente.

Uso tipico:
    python make_splits.py --bench mammo-bench.csv --outdir splits
    python cnn_mammo.py --data-dir splits --base-dir /ruta/a/imagenes --cache-dir cache
"""

import argparse
import hashlib
import math
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
import torchvision.transforms as transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Estadisticas de ImageNet. El backbone viene preentrenado con ellas; usar
# mean=std=0.5 desplaza la entrada respecto a lo que las capas esperan.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def crop_breast(gray):
    """
    Recorta la mama y borra todo lo demas (fondo, etiquetas quemadas, marcadores).

    Los textos y marcadores impresos en la imagen son distintos en cada base de
    datos, asi que son una pista directa del dominio de origen: la red los usa
    como atajo en vez de mirar el tejido. Se conserva unicamente la componente
    conexa mas grande, que siempre es la mama.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return gray, np.ones_like(gray)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 0.02 * gray.size:
        # segmentacion poco fiable: se deja la imagen intacta y sin enmascarar
        return gray, np.ones_like(gray)

    breast = (labels == largest).astype(np.uint8)
    # Cierre morfologico para rellenar huecos internos del tejido denso.
    breast = cv2.morphologyEx(breast, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    x, y, w, h = (stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP],
                  stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT])
    return gray[y:y + h, x:x + w], breast[y:y + h, x:x + w]


def resize_with_padding(gray, target_size):
    """Redimensiona conservando la relacion de aspecto y rellena con negro."""
    th, tw = target_size
    h, w = gray.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((th, tw), dtype=gray.dtype)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess_mammo(path, target_size, do_crop=True, clahe_clip=2.0):
    """
    Pipeline de preprocesado. Notar lo que NO se hace respecto a la version previa:
    ni erosion ni apertura morfologica. Ambas borran estructuras de pocos pixeles,
    que es exactamente el tamano de las microcalcificaciones, el signo radiologico
    mas util para separar maligno de benigno.
    """
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        gray = np.array(Image.open(path).convert("L"))

    mask = None
    if do_crop:
        gray, mask = crop_breast(gray)

    # CLAHE ANTES de enmascarar. Al reves el fondo deja de ser cero: la ecualizacion
    # ve la enorme masa de pixeles en 0 y los desplaza a un valor positivo pequeño
    # (medido: 0 -> 3), asi que el fondo dejaria de poder anularse limpiamente.
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    if mask is not None:
        # El tejido se fuerza a >= 1 para que "pixel == 0" identifique al fondo sin
        # ambiguedad; asi la mascara se puede recuperar de la propia imagen cacheada.
        gray = np.where(mask > 0, np.maximum(gray, 1), 0).astype(np.uint8)

    return resize_with_padding(gray, target_size)


class MammogramDataset(Dataset):
    # Se sube cuando cambia preprocess_mammo, para que un cache generado por una
    # version anterior no se reutilice en silencio con imagenes distintas.
    PREPROC_VERSION = "v2"

    def __init__(self, csv_path, mode="train", base_dir="", img_size=(512, 512),
                 cache_dir=None, do_crop=True, bg_mask=True, path_column="raw_image_path"):
        self.mode = mode
        self.img_size = img_size
        self.cache_dir = cache_dir
        self.do_crop = do_crop
        self.bg_mask = bg_mask

        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        if "label" not in df.columns:
            df["label"] = df["classification"].str.lower().map({"benign": 0, "malignant": 1})
        df = df[df["label"].notna()].copy()

        df["abs_path"] = df[path_column].apply(
            lambda p: p if os.path.isabs(str(p)) else os.path.join(base_dir, str(p))
        )
        missing = (~df["abs_path"].apply(os.path.exists)).sum()
        if missing:
            print(f"  aviso [{mode}]: {missing} imagenes no encontradas, se omiten")
        df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)

        self.df = df
        self.paths = df["abs_path"].values
        self.labels = df["label"].astype(int).values
        self.sources = df["source_dataset"].astype(str).values
        if "patient_key" in df.columns:
            self.patients = df["patient_key"].astype(str).values
        else:
            self.patients = (df["source_dataset"].astype(str) + "|"
                             + df.get("source_subjectid", pd.Series(df.index)).astype(str)).values

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # Las aumentaciones se separan en geometricas y fotometricas porque la
        # mascara de mama tiene que pasar por las primeras (para seguir alineada
        # con el tejido) pero no por las segundas.
        self.geometric = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.11),
                                         antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]) if mode == "train" else None
        self.photometric = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.15, contrast=0.2)], p=0.5
        ) if mode == "train" else None
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.erasing = transforms.RandomErasing(
            p=0.25, scale=(0.02, 0.08), value=0
        ) if mode == "train" else None

    def __len__(self):
        return len(self.labels)

    def _load_gray(self, idx):
        path = self.paths[idx]
        if self.cache_dir:
            # hashlib y no hash(): el hash de strings de Python usa una semilla
            # aleatoria por proceso, asi que cada worker del DataLoader generaria
            # un nombre distinto para la misma imagen y el cache nunca acertaria.
            digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:16]
            cached = os.path.join(
                self.cache_dir,
                f"{digest}_{self.img_size[0]}_{self.PREPROC_VERSION}"
                f"{'_crop' if self.do_crop else ''}.png")
            if os.path.exists(cached):
                img = cv2.imread(cached, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
            img = preprocess_mammo(path, self.img_size, self.do_crop)
            cv2.imwrite(cached, img)
            return img
        return preprocess_mammo(path, self.img_size, self.do_crop)

    def __getitem__(self, idx):
        gray = self._load_gray(idx)
        gray_t = torch.from_numpy(gray)
        # 1 canal replicado a 3: el backbone preentrenado espera RGB.
        img = gray_t.float().div(255.0).unsqueeze(0).repeat(3, 1, 1)
        # El fondo es exactamente 0 gracias a preprocess_mammo, asi que la mascara
        # se recupera de la propia imagen sin guardarla aparte.
        mask = (gray_t > 0).float().unsqueeze(0)

        if self.geometric is not None:
            # Imagen y mascara viajan juntas en un tensor de 4 canales para que la
            # misma transformacion aleatoria se aplique a ambas.
            stacked = self.geometric(torch.cat([img, mask], dim=0))
            img, mask = stacked[:3], stacked[3:4]
            img = self.photometric(img)

        img = self.normalize(img)

        if self.bg_mask:
            # Anular el fondo DESPUES de normalizar es lo que lo deja en 0.0 exacto.
            # Con la normalizacion de ImageNet un pixel negro cae en -2.118, casi
            # 30 veces la magnitud del tejido medio (0.074): la red recibe mas señal
            # del fondo que de la mama. Ademas la silueta del fondo y la cantidad de
            # relleno son distintas en cada base de datos por la relacion de aspecto
            # del detector, o sea otra huella del dominio. En cero el fondo no aporta
            # activacion ni gradiente.
            img = img * (mask > 0.5).float()

        if self.erasing is not None:
            img = self.erasing(img)

        return img, torch.tensor(int(self.labels[idx]), dtype=torch.long)


def make_balanced_sampler(dataset):
    """
    Muestreo que iguala el peso de cada celda (dataset, clase).

    Sin esto, cmmd (78% maligno) y mini-ddsm dominan los lotes y la red aprende
    a inferir la etiqueta desde el estilo de imagen de cada fuente. Igualando las
    celdas, conocer el dataset deja de aportar informacion sobre la clase y el
    atajo se vuelve inutil.
    """
    cells = [f"{s}|{l}" for s, l in zip(dataset.sources, dataset.labels)]
    counts = defaultdict(int)
    for c in cells:
        counts[c] += 1
    weights = torch.tensor([1.0 / counts[c] for c in cells], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        hidden = max(8, in_planes // ratio)
        self.fc1 = nn.Conv2d(in_planes, hidden, 1, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, in_planes, 1, bias=False)

    def forward(self, x):
        avg = self.fc2(self.act(self.fc1(x.mean(dim=(2, 3), keepdim=True))))
        mx = self.fc2(self.act(self.fc1(x.amax(dim=(2, 3), keepdim=True))))
        return torch.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class GatedCBAM(nn.Module):
    """
    CBAM con compuerta residual inicializada en cero.

    Un CBAM clasico insertado en un backbone preentrenado multiplica las
    activaciones por sigmoid(ruido) ~ 0.5 en cada etapa, asi que al empezar el
    entrenamiento destruye justo las features que se querian aprovechar. Con
    gamma=0 el modulo arranca siendo la identidad exacta y la red decide cuanta
    atencion quiere usar en cada etapa.
    """

    def __init__(self, planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(planes, ratio)
        self.sa = SpatialAttention(kernel_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        att = x * self.ca(x)
        att = att * self.sa(att)
        return x + self.gamma * (att - x)


BACKBONES = {
    "convnext_tiny": (models.convnext_tiny, "ConvNeXt_Tiny_Weights"),
    "convnext_small": (models.convnext_small, "ConvNeXt_Small_Weights"),
    "convnext_base": (models.convnext_base, "ConvNeXt_Base_Weights"),
    "convnext_large": (models.convnext_large, "ConvNeXt_Large_Weights"),
}


class ConvNeXtCBAM(nn.Module):
    def __init__(self, arch="convnext_base", num_classes=2, pretrained=True,
                 cbam_stages=2, dropout=0.3, drop_path=0.0, freeze_stages=0):
        super().__init__()
        if arch not in BACKBONES:
            raise ValueError(f"arch debe ser uno de {list(BACKBONES)}")
        fn, weights_name = BACKBONES[arch]
        weights = None
        if pretrained and hasattr(models, weights_name):
            enum = getattr(models, weights_name)
            weights = getattr(enum, "IMAGENET1K_V1", None) or getattr(enum, "DEFAULT", None)
        # stochastic depth: apaga bloques residuales enteros al azar durante el
        # entrenamiento. Es la regularizacion propia de ConvNeXt y actua sobre la
        # profundidad, donde el dropout de la cabeza no llega.
        kwargs = {"stochastic_depth_prob": drop_path} if drop_path > 0 else {}
        self.backbone = fn(weights=weights, **kwargs)

        # Una "etapa" es el Sequential de bloques CNBlock. La deteccion anterior
        # (`len(children) > 1`) tambien capturaba stem y capas de downsample, que
        # son Sequential de 2 elementos, e insertaba CBAM en las 8 posiciones.
        stage_idx = [
            i for i, m in enumerate(self.backbone.features)
            if isinstance(m, nn.Sequential)
            and any("CNBlock" in type(c).__name__ for c in m.children())
        ]
        # Solo las ultimas etapas: ahi viven las features semanticas y el coste
        # de la atencion espacial es bajo porque el mapa ya es pequeno.
        selected = stage_idx[-cbam_stages:] if cbam_stages > 0 else []

        self.cbam = nn.ModuleDict()
        for i in selected:
            planes = self._infer_planes(i)
            self.cbam[str(i)] = GatedCBAM(planes)
        print(f"CBAM insertado en las etapas {selected} de {stage_idx}")

        # Congelar las etapas tempranas: con ~8.6k imagenes, los bordes y texturas
        # genericos de ImageNet no necesitan reajustarse y son capacidad libre para
        # memorizar el conjunto de entrenamiento.
        if freeze_stages > 0:
            for i in range(min(freeze_stages, len(self.backbone.features))):
                for p in self.backbone.features[i].parameters():
                    p.requires_grad_(False)
            print(f"Congeladas las primeras {freeze_stages} capas del backbone")

        in_features = self._infer_planes(len(self.backbone.features) - 1)
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def _infer_planes(self, upto_idx):
        with torch.no_grad():
            x = torch.zeros(1, 3, 128, 128)
            for i, m in enumerate(self.backbone.features):
                x = m(x)
                if i == upto_idx:
                    return x.shape[1]
        raise RuntimeError("indice de etapa fuera de rango")

    def forward(self, x):
        for i, module in enumerate(self.backbone.features):
            x = module(x)
            key = str(i)
            if key in self.cbam:
                x = self.cbam[key](x)
        return self.backbone.classifier(x)


def build_param_groups(model, lr, weight_decay, layer_decay=0.75):
    """
    Learning rate discriminativo por profundidad.

    Las capas tempranas del backbone ya codifican bordes y texturas genericas y
    no necesitan moverse casi nada; la cabeza esta inicializada al azar y necesita
    el LR completo. Ademas se excluyen sesgos y normalizaciones del weight decay,
    donde solo anade ruido.
    """
    n_stages = len(model.backbone.features)
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone.features."):
            depth = int(name.split(".")[2])
            scale = layer_decay ** (n_stages - depth)
        elif name.startswith("cbam.") or name.startswith("backbone.classifier."):
            scale = 1.0  # modulos nuevos: LR completo
        else:
            scale = 1.0
        no_decay = param.ndim <= 1 or name.endswith(".bias") or "gamma" in name
        key = (scale, no_decay)
        groups.setdefault(key, []).append(param)

    return [
        {"params": params, "lr": lr * scale, "weight_decay": 0.0 if nd else weight_decay}
        for (scale, nd), params in groups.items()
    ]


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

class ModelEMA:
    """
    Media movil exponencial de los pesos, con rampa de arranque.

    La rampa es imprescindible: la EMA se inicializa como copia del modelo sin
    entrenar, asi que con decay fijo arrastra esos pesos aleatorios durante miles
    de pasos. Con acumulacion de gradiente el problema se agrava, porque solo hay
    len(loader)/accum actualizaciones por epoca. Validar contra una EMA asi da
    metricas de azar aunque el modelo real vaya bien.

    `min(decay, (1 + n) / (10 + n))` hace que las primeras actualizaciones sigan
    al modelo casi al 100% y el decay suba hacia su valor nominal conforme avanza.
    """

    def __init__(self, model, decay=0.999):
        import copy
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        for ema_p, p in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_p.dtype.is_floating_point:
                ema_p.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                ema_p.copy_(p)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    """
    Coseno sobre TODO el entrenamiento, con calentamiento inicial.

    En la version anterior el scheduler tenia T_max=20 con 35 epocas: el LR
    llegaba a cero en la epoca 20 y despues volvia a subir. Por eso las metricas
    se quedaron congeladas entre la epoca 20 y la 35.
    """
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, fn)


def mixup_batch(images, labels, alpha):
    """
    Mezcla pares de imagenes y sus etiquetas. Impide que la red memorice ejemplos
    concretos, que es justo lo que hace cuando llega a 96% en train y se queda
    en 68% en validacion.
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(images.size(0), device=images.device)
    return lam * images + (1 - lam) * images[perm], labels[perm], lam


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, ema,
                    accum=1, clip=1.0, mixup=0.0):
    """
    Acumulacion de gradiente: con 512x512 no cabe un lote grande en memoria, pero
    acumular `accum` lotes antes de actualizar da el mismo efecto que un batch
    `batch_size * accum` sin gastar mas VRAM. Con batch_size=8 y accum=4 el lote
    efectivo es 32, bastante mas estable que 8.
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for step, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        labels_b, lam = None, 1.0
        if mixup > 0:
            images, labels_b, lam = mixup_batch(images, labels, mixup)

        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if labels_b is not None:
                loss = lam * loss + (1 - lam) * criterion(outputs, labels_b)

        # Se divide entre accum para que el gradiente acumulado sea el promedio
        # del lote efectivo y no su suma (que multiplicaria el LR por accum).
        scaler.scale(loss / accum).backward()

        is_last = (step + 1) == len(loader)
        if (step + 1) % accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema is not None:
                ema.update(model)

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def infer(model, loader, criterion, device, tta=False):
    """Devuelve (loss, probabilidades de malignidad, etiquetas)."""
    model.eval()
    total_loss, total = 0.0, 0
    probs, targets = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
            p = torch.softmax(outputs.float(), dim=1)[:, 1]
            if tta:
                # La mama puede estar en cualquier lateralidad; el espejo horizontal
                # es la unica transformacion que no altera la anatomia.
                p_flip = torch.softmax(model(torch.flip(images, dims=[3])).float(), dim=1)[:, 1]
                p = (p + p_flip) / 2

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        probs.append(p.cpu())
        targets.append(labels.cpu())

    return total_loss / total, torch.cat(probs).numpy(), torch.cat(targets).numpy()


def aggregate_by_patient(probs, y_true, patients, sources=None):
    """
    Colapsa las vistas de un paciente en una sola prediccion.

    Es el nivel al que existe realmente la etiqueta en la mayoria del corpus: en
    cmmd, mini-ddsm y kau-bcmd el 100% de los pacientes con ambas mamas
    fotografiadas comparten etiqueta en las dos, o sea que el diagnostico es por
    paciente y se propago a cada imagen. Como el cancer de mama es unilateral en
    el 95-98% de los casos, eso deja alrededor de la mitad de las imagenes
    "malignas" mostrando una mama sana. Evaluar por imagen mide ese ruido; evaluar
    por paciente mide al modelo.
    """
    agg, lab, src = defaultdict(list), {}, {}
    for i, pk in enumerate(patients):
        agg[pk].append(probs[i])
        lab[pk] = max(lab.get(pk, 0), y_true[i])
        if sources is not None:
            src[pk] = sources[i]
    keys = list(agg)
    p = np.array([np.mean(agg[k]) for k in keys])
    y = np.array([lab[k] for k in keys])
    s = np.array([src[k] for k in keys]) if sources is not None else None
    return p, y, s


def auc_ci(auc, n_pos, n_neg, z=1.96):
    """
    Intervalo de confianza del AUC (Hanley-McNeil).

    Necesario para no leerse como logro un AUC de 1.000 calculado sobre 24 pares:
    con una sola muestra maligna el estimador no tiene ninguna precision.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan"), float("nan")
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = math.sqrt(max(0.0, var))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def best_threshold(y_true, probs, target_sens=None):
    """
    Umbral de trabajo.

    Por defecto maximiza el indice de Youden, que pesa igual sensibilidad y
    especificidad. En cribado de cancer esa no es la compensacion correcta: un
    falso negativo es un tumor que se pasa por alto, y un falso positivo solo
    una prueba adicional. Con target_sens se elige el umbral mas exigente que
    aun alcanza esa sensibilidad.
    """
    fpr, tpr, thr = roc_curve(y_true, probs)
    if target_sens is not None:
        ok = np.where(tpr >= target_sens)[0]
        if len(ok):
            return float(thr[ok[0]])
    return float(thr[np.argmax(tpr - fpr)])


def operating_points(y_true, probs, target_sens=0.85, target_spec=0.90):
    """Puntos de operacion habituales en mamografia, para no reportar solo Youden."""
    fpr, tpr, thr = roc_curve(y_true, probs)
    spec = 1 - fpr
    out = []
    j = int(np.argmax(tpr - fpr))
    out.append(("Youden (max sens+espec)", thr[j], tpr[j], spec[j]))
    ok = np.where(tpr >= target_sens)[0]
    if len(ok):
        i = int(ok[0])
        out.append((f"sensibilidad >= {target_sens:.0%}", thr[i], tpr[i], spec[i]))
    ok = np.where(spec >= target_spec)[0]
    if len(ok):
        i = int(ok[-1])
        out.append((f"especificidad >= {target_spec:.0%}", thr[i], tpr[i], spec[i]))
    return out


def print_operating_points(y_true, probs, title="Puntos de operacion"):
    print(f"\n  {title} (elegidos en validacion):")
    print(f"    {'criterio':<26}{'umbral':>9}{'sens':>8}{'espec':>8}")
    for name, t, se, sp in operating_points(y_true, probs):
        print(f"    {name:<26}{t:>9.3f}{se * 100:>7.1f}%{sp * 100:>7.1f}%")


def report(y_true, probs, threshold, sources=None, patients=None, title="TEST",
           patient_threshold=None):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    print(f"\n{'=' * 62}\n{title}  (umbral = {threshold:.3f})\n{'=' * 62}")
    print(f"AUC-ROC:       {roc_auc_score(y_true, probs):.4f}")
    print(f"Accuracy:      {accuracy_score(y_true, preds) * 100:.2f}%")
    print(f"Recall macro:  {recall_score(y_true, preds, average='macro') * 100:.2f}%")
    print(f"F1 macro:      {f1_score(y_true, preds, average='macro'):.4f}")
    print(f"Sensibilidad:  {tp / max(1, tp + fn) * 100:.2f}%   (malignos detectados)")
    print(f"Especificidad: {tn / max(1, tn + fp) * 100:.2f}%   (benignos correctos)")
    print(f"Matriz: TN={tn} FP={fp} FN={fn} TP={tp}")

    if sources is not None:
        print("\nDesglose por base de datos:")
        print(f"  {'dataset':<14}{'n':>6}{'pares':>8}{'AUC':>7}{'IC 95%':>16}{'Sens':>7}{'Espec':>7}")
        pair_weight, weighted_auc = 0.0, 0.0
        for src in sorted(set(sources)):
            m = sources == src
            yt, pr, pd_ = y_true[m], probs[m], preds[m]
            npos, nneg = int((yt == 1).sum()), int((yt == 0).sum())
            pairs = npos * nneg
            if pairs == 0:
                print(f"  {src:<14}{m.sum():>6}{0:>8}{'n/a':>7}{'una sola clase':>16}")
                continue
            auc = roc_auc_score(yt, pr)
            lo, hi = auc_ci(auc, npos, nneg)
            pair_weight += pairs
            weighted_auc += pairs * auc
            sens = (pd_[yt == 1] == 1).mean() * 100
            spec = (pd_[yt == 0] == 0).mean() * 100
            # Con pocos pares el AUC es ruido: 1.000 sobre 24 pares no dice nada.
            flag = "  <-- muy pocos datos" if pairs < 500 else ""
            print(f"  {src:<14}{m.sum():>6}{pairs:>8}{auc:>7.3f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>16}{sens:>6.1f}%{spec:>6.1f}%{flag}")

        # Descomposicion del AUC agrupado. Los pares que cruzan bases de datos son
        # los que el atajo de dominio resuelve gratis; los pares internos son los
        # unicos que exigen mirar el tejido. Si el AUC cruzado supera al interno,
        # parte del numero de titular lo esta poniendo el dominio, no la lesion.
        total_pairs = int((y_true == 1).sum()) * int((y_true == 0).sum())
        if pair_weight and total_pairs > pair_weight:
            within = weighted_auc / pair_weight
            pooled = roc_auc_score(y_true, probs)
            cross = (pooled * total_pairs - pair_weight * within) / (total_pairs - pair_weight)
            print(f"\n  AUC agrupado (el de titular):        {pooled:.4f}")
            print(f"  AUC DENTRO de cada fuente:           {within:.4f}"
                  f"   <-- la cifra honesta ({pair_weight / total_pairs * 100:.0f}% de los pares)")
            print(f"  AUC entre fuentes distintas:         {cross:.4f}"
                  f"   ({(total_pairs - pair_weight) / total_pairs * 100:.0f}% de los pares)")
            if cross > within + 0.02:
                print("  El AUC cruzado supera al interno: el modelo sigue apoyandose")
                print("  en el dominio de origen. Reporta el AUC interno, no el agrupado.")

    if patients is not None:
        p_pat, y_pat, s_pat = aggregate_by_patient(probs, y_true, patients, sources)
        # Promediar las vistas concentra las probabilidades hacia el centro, asi
        # que el umbral optimo por imagen no lo es por paciente. Se usa uno propio,
        # ajustado tambien en validacion.
        thr_pat = threshold if patient_threshold is None else patient_threshold
        pred_pat = (p_pat >= thr_pat).astype(int)
        npos, nneg = int((y_pat == 1).sum()), int((y_pat == 0).sum())
        auc_pat = roc_auc_score(y_pat, p_pat)
        lo, hi = auc_ci(auc_pat, npos, nneg)
        print(f"\n{'=' * 62}")
        print(f"POR PACIENTE ({len(y_pat)} pacientes, umbral {thr_pat:.3f})"
              "  <-- el nivel al que existe la etiqueta")
        print("=" * 62)
        print(f"AUC {auc_pat:.4f}  IC95 [{lo:.3f}, {hi:.3f}] | "
              f"Acc {accuracy_score(y_pat, pred_pat) * 100:.2f}% | "
              f"F1 macro {f1_score(y_pat, pred_pat, average='macro'):.4f}")

        if s_pat is not None:
            print(f"\n  {'dataset':<14}{'pac.':>6}{'pares':>8}{'AUC':>7}{'IC 95%':>16}")
            wp, wa = 0.0, 0.0
            for source in sorted(set(s_pat)):
                m = s_pat == source
                yt, pr = y_pat[m], p_pat[m]
                a, b = int((yt == 1).sum()), int((yt == 0).sum())
                if a * b == 0:
                    continue
                auc = roc_auc_score(yt, pr)
                l, h = auc_ci(auc, a, b)
                wp += a * b
                wa += a * b * auc
                print(f"  {source:<14}{m.sum():>6}{a * b:>8}{auc:>7.3f}{f'[{l:.3f}, {h:.3f}]':>16}")
            if wp:
                print(f"\n  AUC DENTRO de fuente, por paciente:   {wa / wp:.4f}"
                      "   <-- la cifra a reportar")


def plot_history(history, outdir):
    for metric in ["loss", "acc", "auc"]:
        keys = [k for k in history if k.endswith(metric)]
        if not keys:
            continue
        plt.figure(figsize=(9, 4.5))
        for k in keys:
            plt.plot(history[k], label=k)
        plt.xlabel("epoca")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"curva_{metric}.png"), dpi=120)
        plt.close()


def plot_confusion(y_true, preds, outdir):
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm, cmap="Blues")
    names = ["Benigno", "Maligno"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.title("Matriz de confusion - test")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "matriz_confusion.png"), dpi=120)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="splits", help="carpeta con train/val/test.csv")
    ap.add_argument("--base-dir", default="", help="raiz de las imagenes")
    ap.add_argument("--cache-dir", default=None, help="cachea el preprocesado en disco")
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--arch", default="convnext_base", choices=list(BACKBONES))
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4, help="pasos de acumulacion (batch efectivo)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=0.75)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--cbam-stages", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="stochastic depth en el backbone (0.3-0.5 regulariza fuerte)")
    ap.add_argument("--freeze-stages", type=int, default=0,
                    help="congela las N primeras capas del backbone")
    ap.add_argument("--mixup", type=float, default=0.0,
                    help="alpha de mixup; 0.2-0.4 reduce la memorizacion")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping tras N epocas sin mejorar; 0 lo desactiva")
    ap.add_argument("--select-by", choices=["patient", "image"], default="patient",
                    help="nivel al que se mide el AUC de validacion para elegir el mejor modelo")
    ap.add_argument("--target-sens", type=float, default=0.85,
                    help="sensibilidad minima al fijar el umbral; 0 usa Youden")
    ap.add_argument("--eval-only", action="store_true",
                    help="salta el entrenamiento y evalua el checkpoint ya guardado")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-crop", action="store_true", help="desactiva el recorte de mama")
    ap.add_argument("--no-bg-mask", action="store_true",
                    help="deja el fondo en su valor normalizado (-2.118) en vez de anularlo")
    ap.add_argument("--no-balance", action="store_true", help="desactiva el muestreo balanceado")
    ap.add_argument("--no-ema", action="store_true")
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
    train_ds = MammogramDataset(os.path.join(args.data_dir, "train.csv"), "train", **ds_kwargs)
    val_ds = MammogramDataset(os.path.join(args.data_dir, "val.csv"), "val", **ds_kwargs)
    test_ds = MammogramDataset(os.path.join(args.data_dir, "test.csv"), "test", **ds_kwargs)
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    sampler = None if args.no_balance else make_balanced_sampler(train_ds)
    dl = dict(num_workers=args.workers, pin_memory=True,
              persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              shuffle=sampler is None, drop_last=True, **dl)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, **dl)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **dl)

    model = ConvNeXtCBAM(args.arch, 2, True, args.cbam_stages, args.dropout,
                         drop_path=args.drop_path, freeze_stages=args.freeze_stages).to(device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(build_param_groups(model, args.lr, args.weight_decay, args.layer_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    # El scheduler avanza una vez por actualizacion de pesos, no por lote leido.
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.accum))
    scheduler = cosine_with_warmup(optimizer, warmup_steps=steps_per_epoch,
                                   total_steps=steps_per_epoch * args.epochs)
    ema = None if args.no_ema else ModelEMA(model)

    history = defaultdict(list)
    best_auc, best_epoch = -1.0, -1
    ckpt_path = os.path.join(args.outdir, f"best_{args.arch}_cbam.pth")

    total_epochs = 0 if args.eval_only else args.epochs
    for epoch in range(total_epochs):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          scheduler, device, scaler, ema,
                                          accum=args.accum, mixup=args.mixup)
        # Se evalua el modelo crudo Y la EMA. Comparar los dos hace visible de
        # inmediato cualquier problema de la EMA, en vez de dejar que contamine
        # en silencio toda la seleccion de modelo.
        def evaluate(m):
            loss, probs, true = infer(m, val_loader, criterion, device)
            if args.select_by == "patient":
                # Seleccionar por AUC de paciente alinea el criterio con el nivel
                # al que la etiqueta es correcta; a nivel de imagen, alrededor de
                # la mitad de los "malignos" son la mama sana del paciente.
                p, y, _ = aggregate_by_patient(probs, true, val_ds.patients)
            else:
                p, y = probs, true
            auc = roc_auc_score(y, p) if len(set(y)) > 1 else 0.5
            return loss, auc, accuracy_score(y, (p >= 0.5).astype(int))

        raw_loss, raw_auc, raw_acc = evaluate(model)
        candidates = [("raw", model, raw_loss, raw_auc, raw_acc)]
        if ema is not None:
            e_loss, e_auc, e_acc = evaluate(ema.module)
            candidates.append(("ema", ema.module, e_loss, e_auc, e_acc))

        tag, eval_model, va_loss, va_auc, va_acc = max(candidates, key=lambda c: c[3])

        history["train_loss"].append(tr_loss); history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        history["val_auc"].append(va_auc)

        # El LR util es el de los modulos nuevos (cabeza y CBAM), que es el mayor;
        # param_groups[0] podia ser un grupo profundo del backbone y engañaba.
        lr_now = max(g["lr"] for g in optimizer.param_groups)
        print(f"Epoca {epoch + 1}/{args.epochs}  ({time.time() - t0:.0f}s, lr={lr_now:.2e})")
        print(f"  train  loss {tr_loss:.4f} | acc {tr_acc * 100:.2f}%")
        for name, _, l, a, ac in candidates:
            mark = " <-" if name == tag else ""
            print(f"  val[{name}]  loss {l:.4f} | acc {ac * 100:.2f}% | AUC {a:.4f}{mark}")

        # Seleccion por AUC y no por val_loss: el AUC no depende del umbral ni de
        # la calibracion, que es lo que se degradaba al final del entrenamiento
        # anterior (loss subiendo con accuracy estable = solo sobreconfianza).
        if va_auc > best_auc:
            best_auc, best_epoch = va_auc, epoch
            # float() y no el escalar de numpy: desde PyTorch 2.6 torch.load usa
            # weights_only=True por defecto y rechaza deserializar tipos de numpy.
            torch.save({"epoch": epoch, "model_state_dict": eval_model.state_dict(),
                        "val_auc": float(va_auc), "variant": tag,
                        "args": {k: v for k, v in vars(args).items()}}, ckpt_path)
            print(f"  -> mejor modelo guardado ({tag}, AUC {va_auc:.4f})")
        elif args.patience > 0 and epoch - best_epoch >= args.patience:
            print(f"Early stopping: {args.patience} epocas sin mejorar.")
            break

    if history:
        plot_history(history, args.outdir)

    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No existe el checkpoint {ckpt_path}")
    # weights_only=True es el default desde PyTorch 2.6 y rechaza los escalares de
    # numpy que guardaban las versiones previas de este script. El checkpoint lo
    # genera este mismo codigo, asi que recurrir a weights_only=False es seguro.
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"\nCargando checkpoint: epoca {ckpt.get('epoch', -1) + 1}, "
          f"AUC val {float(ckpt.get('val_auc', float('nan'))):.4f}, "
          f"variante {ckpt.get('variant', 'desconocida')}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # El umbral se elige en validacion, nunca en test.
    _, va_probs, va_true = infer(model, val_loader, criterion, device, tta=True)
    print_operating_points(va_true, va_probs, "Puntos de operacion por imagen")
    thr = best_threshold(va_true, va_probs, target_sens=args.target_sens)
    va_p_pat, va_y_pat, _ = aggregate_by_patient(va_probs, va_true, val_ds.patients)
    print_operating_points(va_y_pat, va_p_pat, "Puntos de operacion por paciente")
    thr_pat = best_threshold(va_y_pat, va_p_pat, target_sens=args.target_sens)
    report(va_true, va_probs, thr, patients=val_ds.patients, title="VALIDACION",
           patient_threshold=thr_pat)

    _, te_probs, te_true = infer(model, test_loader, criterion, device, tta=True)
    report(te_true, te_probs, thr, sources=test_ds.sources,
           patients=test_ds.patients, title="TEST", patient_threshold=thr_pat)
    plot_confusion(te_true, (te_probs >= thr).astype(int), args.outdir)

    pd.DataFrame({"path": test_ds.paths, "source": test_ds.sources,
                  "patient": test_ds.patients, "y_true": te_true,
                  "prob_malignant": te_probs}).to_csv(
        os.path.join(args.outdir, "predicciones_test.csv"), index=False)
    print(f"\nSalidas escritas en {args.outdir}/")


if __name__ == "__main__":
    main()
