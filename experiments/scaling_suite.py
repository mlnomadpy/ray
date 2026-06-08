#!/usr/bin/env python3
"""
Multi-dataset streaming-primal scaling suite for compressed Bernstein--Schur (RAY) features.

STATUS (2026-06): exploratory, NOT cited by any table/figure in main.tex. Superseded as the
in-paper systems result by higgs_scaling.py (tab:higgs) and the appendix scalability figure
(fig:scaling). Retained as an archived reproducible artifact; it remains the cleanest
demonstration that at d=5000 the exact-modulation feature (d_b ~ 1.25e7) cannot be built and
only sketched RAY runs (direct proof of Limitation (v)). Candidate for a future high-d table.

Generalizes higgs_scaling.py across a spread of standard binary-classification datasets that
stress the two axes the paper cares about:

  (A) LARGE N  -> exact Gram impossible; can we train as a memory-flat streaming primal?
  (B) HIGH d   -> d_b = d(d+1)/2+d+1 = O(d^2) blows up; exact RAY becomes IMPOSSIBLE to even
                  build (one feature vector is d_b floats), so ONLY TensorSketch-RAY runs.
                  This is the experiment that converts TS-RAY from "nice compression" to
                  "the only option", directly proving the d_b floor argument (Limitation v).

It reuses the MLX feature builders + streaming Adam trainer from higgs_scaling.py unchanged,
so results are directly comparable. The only additions here are: (i) a dataset registry with
download manifest, (ii) a LIBSVM/CSV loader that densifies + off-sphere-rescales like HIGGS,
(iii) a guard that SKIPS exact RAY when d_b exceeds --ray-cap (reported as "exact RAY: N/A").

Datasets (label col -> {0,1}; off-sphere: standardize then /= 99.9th norm percentile):

  name        N_train   N_test    d      d_b         axis                  exact RAY?
  ---------   -------   -------   ----   ---------   -------------------   ----------
  higgs       10.5M     500k       28        435     large N                yes
  susy         4.5M     500k       18        190     large N + coupling?    yes
  hepmass      7.0M     3.5M       27        406     large N + coupling?    yes
  miniboone    104k      26k       50      1,326     fast physics probe     yes
  covtype      ~522k     ~58k      54      1,540     mid N / mid d          yes
  a9a          32.5k     16.3k    123      7,626     tabular sanity         yes (heavy)
  madelon       2.0k      1.8k    500    125,751     nonlinear/XOR          NO (d_b>cap)
  epsilon       400k      100k   2000  2,003,000     high d + large N       NO (d_b>cap)
  gisette       6.0k      1.0k   5000 12,512,500     extreme high d         NO (d_b>cap)

Two axes. FLOOR/SCALABILITY (TS-RAY is NECESSARY): gisette/epsilon/madelon force d_b=O(d^2) so exact
RAY cannot be built; susy/hepmass/covtype force the streaming primal. COUPLING (RAY may WIN): physics
signal/background (susy/hepmass/miniboone) and gisette/madelon's degree-2 interaction structure (pixel
products / hypercube-XOR) are real-data proxies for the alignment x proximity story. Run gate_diagnostic.py
first (cheap, 50k sample) to see which datasets actually have the coupling before spending training compute.
Frozen vision embeddings (CIFAR/ImageNet via CLIP/DINOv2) are the advisor's top coupling bet -- separate harness.

Env: /opt/homebrew/bin/python3 (mlx, numpy, scipy, sklearn; pandas only for higgs CSV).
Data dir: ~/higgs_data/ (HIGGS.csv.gz) and ~/rf_data/ (LIBSVM files; auto-downloaded if absent).

Examples:
  # validate the high-d floor story on the two small high-d sets (fast, no big download):
  /opt/homebrew/bin/python3 scaling_suite.py --datasets gisette madelon --Ms 512 2048 8192
  # the large-N companions to HIGGS:
  /opt/homebrew/bin/python3 scaling_suite.py --datasets susy covtype --Ms 512 1024 2048 4096 8192
  # everything (downloads ~12GB for epsilon; run deliberately):
  /opt/homebrew/bin/python3 scaling_suite.py --datasets all --Ms 512 1024 2048 4096 8192
"""
import argparse, os, time, json, subprocess
import numpy as np

# reuse the exact same feature builders + trainer as the HIGGS experiment
from higgs_scaling import make_params, build, train, peak_gb, log, T0  # noqa: F401

HERE = os.path.dirname(__file__)
DATA = os.path.expanduser("~/rf_data")
HIGGS_PATH = os.path.expanduser("~/higgs_data/HIGGS.csv.gz")
LIBSVM = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

# Two selection axes. FLOOR/SCALABILITY: high-d forces d_b=O(d^2) so exact RAY is impossible and
# only TS-RAY runs (gisette/epsilon/madelon); large-N forces the streaming primal (susy/covtype/
# hepmass). COUPLING (where RAY should WIN): targets that need local proximity x directional
# alignment -> physics signal/background (susy/hepmass/miniboone) and frozen vision embeddings
# (cifar/imagenet, separate harness). Run gate_diagnostic.py first to see which have the structure.
#
# fmt: "csv0" label in col 0 (higgs/hepmass) | "libsvm" sparse (.bz2 ok) | "miniboone" UCI text.
REGISTRY = {
    "higgs":     {"fmt": "csv0", "path": HIGGS_PATH, "feat": (1, 29), "header": None, "ntr": 10_500_000, "nte": 500_000},
    "susy":      {"fmt": "csv0", "tr": UCI + "00279/SUSY.csv.gz", "feat": (1, 19), "header": None,
                  "ntr": 4_500_000, "nte": 500_000},
    "hepmass":   {"fmt": "csv0", "tr": UCI + "00347/all_train.csv.gz", "te": UCI + "00347/all_test.csv.gz",
                  "feat": (1, 28), "header": 0, "ntr": 7_000_000, "nte": 3_500_000},
    "miniboone": {"fmt": "miniboone", "url": UCI + "00199/MiniBooNE_PID.txt", "ntr": 104_000, "nte": 26_000},
    "covtype":   {"fmt": "libsvm", "tr": LIBSVM + "covtype.libsvm.binary.scale.bz2", "te": None,
                  "ntr": 522_000, "nte": 58_000},
    "a9a":       {"fmt": "libsvm", "tr": LIBSVM + "a9a", "te": LIBSVM + "a9a.t", "ntr": 32_561, "nte": 16_281},
    "madelon":   {"fmt": "libsvm", "tr": LIBSVM + "madelon", "te": LIBSVM + "madelon.t", "ntr": 2_000, "nte": 1_800},
    "epsilon":   {"fmt": "libsvm", "tr": LIBSVM + "epsilon_normalized.bz2", "te": LIBSVM + "epsilon_normalized.t.bz2",
                  "ntr": 400_000, "nte": 100_000},
    "gisette":   {"fmt": "libsvm", "tr": LIBSVM + "gisette_scale.bz2", "te": LIBSVM + "gisette_scale.t.bz2",
                  "ntr": 6_000, "nte": 1_000},
}


def _fetch(url):
    if url is None:
        return None
    os.makedirs(DATA, exist_ok=True)
    dst = os.path.join(DATA, os.path.basename(url))
    if not os.path.exists(dst):
        log(f"downloading {url} -> {dst}")
        # curl uses the macOS system cert store (Python 3.14 + LIBSVM cert chain fails ssl verify)
        tmp = dst + ".part"
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", tmp, url], check=True)
        os.replace(tmp, dst)
    return dst


def _to01(y):
    y = np.asarray(y, np.float64)
    if y.size == 0:
        return y.astype(np.float32)
    return (y > np.median(y)).astype(np.float32) if len(np.unique(y)) > 2 else (y == y.max()).astype(np.float32)


def _offsphere(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    scale = np.percentile(np.linalg.norm(Xtr, axis=1), 99.9) + 1e-9   # bounded-ball, VARYING norms
    return (Xtr / scale).astype(np.float32), (Xte / scale).astype(np.float32)


def _load_raw(name, cap=None):
    """Return (Xtr, ytr, Xte, yte) in {0,1}, before subsampling/normalization.
    cap (if set) limits rows read from dense CSVs -- lets the gate diagnostic subsample
    a 5M-row file without parsing all of it (the download still happens once, then caches)."""
    r = REGISTRY[name]
    if r["fmt"] == "csv0":                                   # label in column 0, dense CSV (higgs/hepmass/susy)
        import pandas as pd
        c0, c1 = r["feat"]; hdr = r.get("header", None)
        if r.get("te"):                                      # separate train/test files (hepmass)
            ntr = min(r["ntr"], cap) if cap else r["ntr"]
            dtr = pd.read_csv(_fetch(r["tr"]), header=hdr, dtype=np.float32, nrows=ntr)
            dte = pd.read_csv(_fetch(r["te"]), header=hdr, dtype=np.float32, nrows=r["nte"])
            return (dtr.iloc[:, c0:c1].to_numpy(np.float32), _to01(dtr.iloc[:, 0].to_numpy(np.float32)),
                    dte.iloc[:, c0:c1].to_numpy(np.float32), _to01(dte.iloc[:, 0].to_numpy(np.float32)))
        path = r.get("path") or _fetch(r["tr"])              # single file (higgs/susy): first ntr train, next nte test
        rows = min(r["ntr"] + r["nte"], cap) if cap else r["ntr"] + r["nte"]
        df = pd.read_csv(path, header=hdr, dtype=np.float32, nrows=rows)
        n_tr = min(r["ntr"], len(df))                        # if capped below ntr, all rows are train (test empty)
        y = df.iloc[:, 0].to_numpy(np.float32); X = df.iloc[:, c0:c1].to_numpy(np.float32)
        return X[:n_tr], _to01(y[:n_tr]), X[n_tr:], _to01(y[n_tr:])
    if r["fmt"] == "miniboone":                              # UCI text: header "Nsig Nbg", then sig rows, then bg rows
        path = _fetch(r["url"])
        with open(path) as f:
            nsig, nbg = (int(v) for v in f.readline().split())
        arr = np.loadtxt(path, skiprows=1, dtype=np.float32)
        arr[arr <= -999] = np.nan                            # -999 = unmeasured sentinel; impute col mean
        col_mean = np.nanmean(arr, axis=0)
        arr = np.where(np.isnan(arr), col_mean[None, :], arr).astype(np.float32)
        y = np.concatenate([np.ones(nsig, np.float32), np.zeros(nbg, np.float32)])
        rng = np.random.default_rng(0); perm = rng.permutation(len(y))
        arr, y = arr[perm], y[perm]
        return arr[r["nte"]:], y[r["nte"]:], arr[:r["nte"]], y[:r["nte"]]
    from sklearn.datasets import load_svmlight_file                                # libsvm sparse
    Xtr_s, ytr = load_svmlight_file(_fetch(r["tr"]))
    Xtr = np.asarray(Xtr_s.todense(), np.float32); ytr = _to01(ytr)
    if r["te"] is not None:
        Xte_s, yte = load_svmlight_file(_fetch(r["te"]), n_features=Xtr.shape[1])
        return Xtr, ytr, np.asarray(Xte_s.todense(), np.float32), _to01(yte)
    return Xtr[r["nte"]:], ytr[r["nte"]:], Xtr[:r["nte"]], ytr[:r["nte"]]   # single file: last nte = test


def load(name, n_train, n_test):
    r = REGISTRY[name]
    cap = (n_train + n_test) if n_train + n_test < 5_000_000 else None   # fast path for the diagnostic
    Xtr, ytr, Xte, yte = _load_raw(name, cap)
    n_train = min(n_train, r["ntr"], len(ytr)); n_test = min(n_test, r["nte"], len(yte))
    rng = np.random.default_rng(0); perm = rng.permutation(len(ytr))[:n_train]
    Xtr, ytr = Xtr[perm], ytr[perm]; Xte, yte = Xte[:n_test], yte[:n_test]
    Xtr, Xte = _offsphere(Xtr, Xte)
    return Xtr, ytr, Xte, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["gisette", "madelon"])
    ap.add_argument("--n-train", type=int, default=10**9)
    ap.add_argument("--n-test", type=int, default=10**9)
    ap.add_argument("--Ms", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--m-sketch", type=int, default=128)
    ap.add_argument("--ray-cap", type=int, default=20000, help="skip exact RAY when d_b exceeds this")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--target-steps", type=int, default=3000,
                    help="min total SGD steps; small datasets get more epochs so RF methods actually train")
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--lam", type=float, default=1e-5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "scaling_suite.json"))
    args = ap.parse_args()
    names = list(REGISTRY) if args.datasets == ["all"] else args.datasets
    out = {"config": vars(args), "datasets": {}}
    for name in names:
        log(f"==== {name} ====")
        Xtr, ytr, Xte, yte = load(name, args.n_train, args.n_test)
        d = Xtr.shape[1]
        d_b = d * (d + 1) // 2 + d + 1
        # eps = median squared pairwise distance on a small sample
        s = Xtr[:2000]
        eps = float(np.median(np.sum((s[:, None] - s[None]) ** 2, -1)[np.triu_indices(len(s), 1)]))
        gamma = 1.0 / eps; b = 1.0
        ray_ok = d_b <= args.ray_cap
        # adapt batch/epochs so RF methods actually converge on small datasets (bs=8192 on N=2k = 1 batch)
        bs = min(args.bs, max(64, len(ytr) // 8))
        spe = max(1, (len(ytr) + bs - 1) // bs)
        epochs = max(args.epochs, min(300, -(-args.target_steps // spe)))
        log(f"N_train={len(ytr):,} N_test={len(yte):,} d={d} d_b={d_b} eps={eps:.3f} "
            f"pos_rate={ytr.mean():.3f} exactRAY={'yes' if ray_ok else 'N/A (d_b>cap)'} "
            f"bs={bs} epochs={epochs} (~{epochs*spe} steps)")
        methods = ["linear", "gauss", "tsray"] + (["ray"] if ray_ok else [])
        rows = []
        jobs = [("linear", d)] + [(m, M) for M in args.Ms for m in methods if m not in ("linear",)]
        for method, Mtarget in jobs:
            p = make_params(method, d, Mtarget, eps, gamma, b, args.m_sketch, 0)
            t0 = time.time()
            auc, ll, tb = train(Xtr, ytr, Xte, yte, p, epochs, bs, args.lr, args.lam)
            rows.append({"method": method, "M_target": Mtarget, "M_actual": p["M"], "D": p.get("D"),
                         "auc": auc, "log_loss": ll, "train_wall_s": time.time() - t0,
                         "build_wall_s": tb, "peak_gb": peak_gb()})
            log(f"  {method:6s} M={p['M']:7d} D={str(p.get('D')):>4}  AUC={auc:.4f}  "
                f"logloss={ll:.4f}  peak={peak_gb():.1f}GB")
        out["datasets"][name] = {"d": d, "d_b": d_b, "eps": eps, "n_train": int(len(ytr)),
                                 "n_test": int(len(yte)), "pos_rate": float(ytr.mean()),
                                 "exact_ray": ray_ok, "rows": rows}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
