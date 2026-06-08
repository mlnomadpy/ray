#!/usr/bin/env python3
"""
Extract frozen CLIP (ViT-B/32) embeddings for CIFAR-10/100 -> .npz (X, y), for the coupling track.

Hypothesis (advisor): in a contrastively-trained embedding space, class identity is ANGULAR
(alignment) while local neighborhoods are PROXIMITY, so the alignment-gate x local-proximity
product the biased yat-kernel computes should help where a radial- or alignment-only kernel
each false-positives. CLIP is chosen because it is cosine-trained, so its geometry is angular.

This script only builds the embeddings. Then:
  gate_diagnostic.py --npz <out>.npz --name cifar10-clip     # cheap pre-flight: is coupling present?
and only if the gate says yes do we build the multiclass training harness.

No PIL/torchvision needed: CIFAR raw (cs.toronto.edu) -> numpy; resize 32->224 with torch
F.interpolate (bicubic); CLIP normalize; CLIPModel.get_image_features on MPS.

Env: /opt/homebrew/bin/python3 (torch 2.11 + MPS, transformers). Data dir: ~/rf_data/.
Run: /opt/homebrew/bin/python3 cifar_embed.py --dataset cifar10 --out ~/rf_data/cifar10_clip.npz
"""
import argparse, os, time, tarfile, pickle, subprocess
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
DATA = os.path.expanduser("~/rf_data")
URLS = {
    "cifar10":  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    "cifar100": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
}


def _fetch(url):
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, os.path.basename(url))
    if not os.path.exists(dst):
        log(f"downloading {url}")
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", dst + ".part", url], check=True)
        os.replace(dst + ".part", dst)
    return dst


def _load_cifar(dataset):
    """Return (Xtr uint8 (N,32,32,3), ytr, Xte, yte). Pure numpy, no PIL."""
    tar = _fetch(URLS[dataset])
    with tarfile.open(tar) as t:
        members = {m.name.split("/")[-1]: m for m in t.getmembers() if m.isfile()}

        def rd(name):
            return pickle.load(t.extractfile(members[name]), encoding="bytes")
        if dataset == "cifar10":
            tr = [rd(f"data_batch_{i}") for i in range(1, 6)]
            Xtr = np.concatenate([b[b"data"] for b in tr]); ytr = np.concatenate([b[b"labels"] for b in tr])
            te = rd("test_batch"); Xte = te[b"data"]; yte = np.array(te[b"labels"])
        else:
            tr = rd("train"); Xtr = tr[b"data"]; ytr = np.array(tr[b"fine_labels"])
            te = rd("test"); Xte = te[b"data"]; yte = np.array(te[b"fine_labels"])
    rs = lambda X: X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # NCHW-flat -> NHWC
    return rs(Xtr), ytr.astype(np.int64), rs(Xte), yte.astype(np.int64)


def embed(Xu8, model, device, bs):
    import torch
    import torch.nn.functional as F
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(Xu8), bs):
        x = torch.from_numpy(Xu8[i:i + bs]).to(device).float().div_(255.0).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():                                # explicit projected path (version-robust)
            pooled = model.vision_model(pixel_values=x).pooler_output
            emb = model.visual_projection(pooled)            # -> 512-d cosine-space embedding
        out.append(emb.float().cpu().numpy())
        if (i // bs) % 20 == 0:
            log(f"  embedded {i+len(x):,}/{len(Xu8):,}")
    return np.concatenate(out).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = os.path.expanduser(args.out or os.path.join(DATA, f"{args.dataset}_clip.npz"))
    import torch
    from transformers import CLIPModel
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading {args.model} on {device}")
    model = CLIPModel.from_pretrained(args.model).to(device).eval()
    Xtr, ytr, Xte, yte = _load_cifar(args.dataset)
    log(f"{args.dataset}: train {Xtr.shape} test {Xte.shape}, {len(np.unique(ytr))} classes")
    Etr = embed(Xtr, model, device, args.bs); log(f"train embedded {Etr.shape}")
    Ete = embed(Xte, model, device, args.bs); log(f"test embedded {Ete.shape}")
    # save combined (gate_diagnostic uses X,y on train); keep splits for the trainer
    np.savez(out, X=Etr, y=ytr, X_test=Ete, y_test=yte)
    log(f"wrote {out}  (||e|| mean {np.linalg.norm(Etr,axis=1).mean():.2f})")


if __name__ == "__main__":
    main()
