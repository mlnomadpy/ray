#!/usr/bin/env python3
"""
QM9 cache builder: Coulomb-matrix eigenspectrum descriptor + atomization energies.

Parses gdb9.sdf (DeepChem mirror of QM9, 133,885 molecules; V2000 SDF, no rdkit
needed) and gdb9.sdf.csv (properties; u0_atom = U0 atomization energy already
referenced against single atoms). Descriptor is the SIZE-EXTENSIVE Coulomb-matrix
eigenspectrum of Rupp et al. (2012):
    C_ii = 0.5 Z_i^2.4,  C_ij = Z_i Z_j / |R_i - R_j|   (R in Angstrom),
eigenvalues sorted by |.| descending, zero-padded to 29 (max atoms in QM9).
The norm of this descriptor grows with molecule size (extensivity), the direction
carries composition/geometry -- the alignment x proximity coupling under test.

Also caches element counts (H,C,N,O,F) for the dressed-atom (composition-linear)
baseline. The 3,054 uncharacterized molecules are NOT excluded (the figshare list
is WAF-blocked); they are 2.3% of the data and identical across all compared
kernels, so the comparison is unaffected.

------------------------------------------------------------------------------
REPRODUCIBILITY
    Data: ~/rf_data/qm9/gdb9.tar.gz from
          https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
    Env : python3>=3.9, numpy. CPU, ~2 min.
    Run : ~/.pixi/envs/jax/bin/python3 qm9_build_cache.py
    Out : ~/rf_data/qm9/qm9_cm.npz  (X (N,29) f64, y_u0atom (N,) kcal/mol if csv
          is kcal/mol -- magnitude printed for the unit sanity check, counts (N,5),
          natoms (N,), ids (N,))
------------------------------------------------------------------------------
"""
import os, tarfile, time, io, csv
import numpy as np

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

QM9 = os.path.expanduser("~/rf_data/qm9")
Z = {"H": 1.0, "C": 6.0, "N": 7.0, "O": 8.0, "F": 9.0}
ELEMS = ["H", "C", "N", "O", "F"]
DMAX = 29


def cm(coords, zs):
    z = np.array(zs)
    C = np.outer(z, z)
    R = np.array(coords)
    d = np.sqrt(np.maximum(((R[:, None] - R[None]) ** 2).sum(-1), 1e-12))
    np.fill_diagonal(d, 1.0)
    C = C / d
    np.fill_diagonal(C, 0.5 * z ** 2.4)
    return C


def cm_eigs(coords, zs):
    ev = np.linalg.eigvalsh(cm(coords, zs))
    ev = ev[np.argsort(-np.abs(ev))]
    out = np.zeros(DMAX)
    out[: len(ev)] = ev
    return out


IU = np.triu_indices(DMAX)


def cm_full(coords, zs):
    """Sorted Coulomb matrix (rows/cols by row norm, descending), padded upper triangle."""
    C = cm(coords, zs)
    order = np.argsort(-np.linalg.norm(C, axis=1))
    C = C[np.ix_(order, order)]
    P = np.zeros((DMAX, DMAX))
    n = C.shape[0]
    P[:n, :n] = C
    return P[IU]


def parse_sdf(stream):
    """Yield (name, coords, elems) per molecule from a V2000 SDF text stream."""
    while True:
        name = stream.readline()
        if not name:
            return
        name = name.strip()
        stream.readline(); stream.readline()           # comment lines
        counts = stream.readline()
        if not counts:
            return
        na = int(counts[:3])
        nb = int(counts[3:6])
        coords, elems = [], []
        for _ in range(na):
            parts = stream.readline().split()
            coords.append((float(parts[0]), float(parts[1]), float(parts[2])))
            elems.append(parts[3])
        for _ in range(nb):
            stream.readline()
        while True:                                     # skip to record terminator
            line = stream.readline()
            if not line or line.startswith("$$$$"):
                break
        yield name, coords, elems


def main():
    log("reading gdb9.sdf.csv ...")
    props = {}
    with tarfile.open(os.path.join(QM9, "gdb9.tar.gz")) as tf:
        f = tf.extractfile("gdb9.sdf.csv")
        rdr = csv.DictReader(io.TextIOWrapper(f))
        for row in rdr:
            props[row["mol_id"]] = float(row["u0_atom"])
        log(f"  {len(props):,} property rows; u0_atom sample: "
            f"{list(props.values())[:3]}")
        log("parsing gdb9.sdf ...")
        f = tf.extractfile("gdb9.sdf")
        stream = io.TextIOWrapper(f)
        X, Xf, y, counts, natoms, ids = [], [], [], [], [], []
        skipped = 0
        for i, (name, coords, elems) in enumerate(parse_sdf(stream)):
            if name not in props or any(e not in Z for e in elems):
                skipped += 1
                continue
            zlist = [Z[e] for e in elems]
            X.append(cm_eigs(coords, zlist))
            Xf.append(cm_full(coords, zlist).astype(np.float32))
            y.append(props[name])
            counts.append([elems.count(e) for e in ELEMS])
            natoms.append(len(elems))
            ids.append(name)
            if (i + 1) % 20000 == 0:
                log(f"  {i+1:,} molecules ...")
    X = np.array(X); y = np.array(y)
    log(f"done: N={len(X):,} (skipped {skipped}); d={X.shape[1]}")
    log(f"unit sanity: y mean={y.mean():.1f}, std={y.std():.1f}, range=[{y.min():.1f},{y.max():.1f}]"
        f"  (QM9 U0_atom in kcal/mol is ~ -1100 +- 200)")
    log(f"descriptor norms: median={np.median(np.linalg.norm(X,axis=1)):.1f}, "
        f"q99={np.percentile(np.linalg.norm(X,axis=1),99):.1f} (extensivity: grows with natoms, "
        f"corr={np.corrcoef(np.linalg.norm(X,axis=1), natoms)[0,1]:.3f})")
    np.savez_compressed(os.path.join(QM9, "qm9_cm.npz"), X=X, y=y,
                        counts=np.array(counts), natoms=np.array(natoms),
                        ids=np.array(ids))
    log(f"wrote {os.path.join(QM9,'qm9_cm.npz')}")
    Xf = np.array(Xf)
    log(f"full sorted CM: d={Xf.shape[1]}")
    np.savez_compressed(os.path.join(QM9, "qm9_cmfull.npz"), X=Xf, y=y,
                        counts=np.array(counts), natoms=np.array(natoms),
                        ids=np.array(ids))
    log(f"wrote {os.path.join(QM9,'qm9_cmfull.npz')}")


if __name__ == "__main__":
    main()
