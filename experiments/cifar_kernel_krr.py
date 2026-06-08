#!/usr/bin/env python3
"""
EXACT-kernel KRR on CIFAR CLIP embeddings: is the yat-kernel no-regret across regimes as a KERNEL,
separate from its random-feature approximation? (cifar_classify.py showed the RF approximation
yat-RAY underperforms on alignment-dominated data; this tests the exact kernel the user's claim
is actually about.)

For each kernel we form the NxN Gram on a subsample, fit closed-form KRR on one-hot labels
(alpha = (K+lam I)^-1 Y), and report test top-1. We run TWO preprocessings of the same embeddings
to exhibit both regimes:
  - standardized (centered)  -> ALIGNMENT-dominated  (poly should win, radial should lag)
  - raw bounded-ball         -> PROXIMITY-dominated   (radial should win, poly should lag)
The no-regret claim: yat is near the top in BOTH, while each single-factor kernel collapses in one.

Kernels: gaussian exp(-g D2); imq 1/(D2+eps); poly (x.w+b)^2; yat (x.w+b)^2/(D2+eps);
yatN normalized yat.  D2 = squared distance, s = x.w.

Env: /opt/homebrew/bin/python3 (numpy). Embeddings: ~/rf_data/{cifar10,cifar100}_clip.npz.
Run: cifar_kernel_krr.py --datasets cifar10 cifar100 --n 10000 --nte 2000

REPRODUCIBILITY (results in main.tex Table tab:cifar, sec:exp_necessity). Test acc, lam=0.1,
  5 seeds (random 10k/2k subsamples), mean+-std (std<=0.006):
  cifar10  std(align) gaussian 0.944 imq 0.944 poly 0.944 yat 0.944   (4-way tie)
  cifar10  raw(prox)  gaussian 0.942 imq 0.942 poly 0.944 yat 0.942
  cifar100 std(align) gaussian 0.768 imq 0.766 poly 0.771 yat 0.766
  cifar100 raw(prox)  gaussian 0.763 imq 0.760 poly 0.768 yat 0.761
  Headline: all 4 kernels agree within ~1 std; yat competitive in both regimes (never far
  from best), poly leads CIFAR-100 by ~1 std. No kernel separates decisively on CLIP embeddings.
  (RF approximation yat-RAY needs more draws at d=512: cifar_rf_fair.py.)
"""
import argparse, os, time, json
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
HERE = os.path.dirname(__file__)


def prep(X, mode):
    X = X.astype(np.float64)
    if mode == "std":
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    else:                                  # raw bounded-ball: keep direction+norm, just bound the radius
        X = X - X.mean(0)
    X = X / (np.percentile(np.linalg.norm(X, axis=1), 99.9) + 1e-9)
    return X


def grams(Xa, Xb, eps, b):
    """Return dict of (na x nb) Gram matrices for each kernel."""
    S = Xa @ Xb.T
    na = (Xa * Xa).sum(1)[:, None]; nb = (Xb * Xb).sum(1)[None, :]
    D2 = np.maximum(na + nb - 2 * S, 0.0)
    g = 1.0 / eps
    P = (S + b) ** 2
    H = 1.0 / (D2 + eps)
    K = {"gaussian": np.exp(-g * D2), "imq": H, "poly": P, "yat": P * H}
    norm = ((na + b) ** 0.5) @ ((nb + b) ** 0.5).T if False else np.sqrt((na + b)) * np.sqrt((nb + b))
    K["yatN"] = ((S + b) ** 2 / (norm ** 2)) * H * np.maximum(na.max(), 1.0)  # bounded gate * radial
    return K


def krr_acc(Ktr, ytr, Kte, yte, C, lam):
    Y = np.zeros((len(ytr), C)); Y[np.arange(len(ytr)), ytr] = 1.0
    alpha = np.linalg.solve(Ktr + lam * np.eye(len(ytr)), Y)
    return float((np.argmax(Kte @ alpha, 1) == yte).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--nte", type=int, default=2000)
    ap.add_argument("--lam", type=float, default=1e-1)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "cifar_kernel_krr.json"))
    args = ap.parse_args()
    out = {"config": vars(args), "results": {}}
    for name in args.datasets:
        z = np.load(os.path.expanduser(f"~/rf_data/{name}_clip.npz"))
        for mode in ["std", "raw"]:
            per_seed = {}
            for sd in range(args.seeds):                      # independent train/test subsamples -> mean+-std
                rng = np.random.default_rng(sd)
                itr = rng.permutation(len(z["y"]))[:args.n]; ite = rng.permutation(len(z["y_test"]))[:args.nte]
                ytr = z["y"][itr].astype(int); yte = z["y_test"][ite].astype(int); C = int(ytr.max()) + 1
                Xtr = prep(z["X"][itr], mode); Xte = prep(z["X_test"][ite], mode)
                s = Xtr[:2000]
                eps = float(np.median(np.sum((s[:, None] - s[None]) ** 2, -1)[np.triu_indices(len(s), 1)]))
                Ktr = grams(Xtr, Xtr, eps, 1.0); Kte = grams(Xte, Xtr, eps, 1.0)
                for k in Ktr:
                    per_seed.setdefault(k, []).append(krr_acc(Ktr[k], ytr, Kte[k], yte, C, args.lam))
            mean = {k: float(np.mean(v)) for k, v in per_seed.items()}
            std = {k: float(np.std(v)) for k, v in per_seed.items()}
            best = max(mean, key=mean.get)
            log(f"{name:8s} {mode:3s}  " + "  ".join(f"{k}={mean[k]:.4f}+-{std[k]:.4f}" for k in mean)
                + f"   [best={best}]")
            out["results"][f"{name}-{mode}"] = {"acc": mean, "std": std, "best": best}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
