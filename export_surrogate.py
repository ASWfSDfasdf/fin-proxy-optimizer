#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""export_surrogate.py —— 把 3D 单肋 RBF 代理模型导出成可独立运行的包。

输入：results.csv（4096 行实测，12 列）
输出：
  surrogate.npz     中心点(float32) + 两套权重 + 多项式系数 + 归一化常数 + meta
  cv_metrics.csv    10 折 CV 指标（degree=0 / degree=1 两版对比）
  table_Lt_Q.csv    h=200,k=400 的 8x8 (L,t)->Q 等值表
  export_report.txt 自检记录

不调用 fin3d.solve()，不重跑仿真，只读 results.csv。
"""
import os, sys, csv, json, time
import numpy as np
from scipy.interpolate import RBFInterpolator

BASE = "/work/home/ac0gmvgc4o/fin3d/scan_20261115"
CSV = os.path.join(BASE, "results.csv")
OUT = os.path.join(BASE, "analysis")
os.makedirs(OUT, exist_ok=True)

LNAMES = ["L_mm", "t_mm", "h", "k"]
BOUNDS = {"L_mm": (30.0, 100.0), "t_mm": (1.0, 5.0), "h": (10.0, 200.0), "k": (50.0, 400.0)}
SEED = 20261115
NFOLD = 10


def load():
    rows = []
    with open(CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    X = np.array([[float(r["L_mm"]), float(r["t_mm"]), float(r["h"]), float(r["k"])] for r in rows])
    Q = np.array([float(r["Q_W_per_m"]) for r in rows])
    T = np.array([float(r["tip_T_K"]) for r in rows])
    cid = np.array([int(r["case_id"]) for r in rows], dtype=np.int64)
    return X, Q, T, cid


def phi_from_r2(r2):
    """phi(r) = r^2 * ln r  ==  0.5 * r2 * ln(r2)，r=0 处取 0。"""
    out = np.zeros_like(r2)
    m = r2 > 0.0
    out[m] = 0.5 * r2[m] * np.log(r2[m])
    return out


def phi_block(Xq, C, chunk=512):
    n = Xq.shape[0]
    out = np.empty((n, C.shape[0]))
    for i in range(0, n, chunk):
        d = Xq[i:i + chunk, None, :] - C[None, :, :]
        out[i:i + chunk] = phi_from_r2(np.einsum("ijk,ijk->ij", d, d))
    return out


def poly_terms(X, degree):
    n = X.shape[0]
    if degree == 0:
        return np.ones((n, 1))
    if degree == 1:
        return np.hstack([np.ones((n, 1)), X])
    raise ValueError(degree)


def fit_tps(Xn, y, degree):
    n = Xn.shape[0]
    K = phi_block(Xn, Xn)
    P = poly_terms(Xn, degree)
    m = P.shape[1]
    A = np.zeros((n + m, n + m))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    sol = np.linalg.solve(A, np.concatenate([y, np.zeros(m)]))
    return sol[:n], sol[n:]


def pred_tps(Xq, C, w, poly, degree):
    P = poly_terms(Xq, degree)
    return phi_block(Xq, C) @ w + P @ poly


def metrics(y_true, y_pred, target):
    if target == "lnQ":
        err = np.abs(np.exp(y_pred) - np.exp(y_true)) / np.exp(y_true)
        r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        r2Q = 1.0 - np.sum((np.exp(y_pred) - np.exp(y_true)) ** 2) / np.sum((np.exp(y_true) - np.exp(y_true).mean()) ** 2)
        return dict(r2_logQ=r2, r2_Q=r2Q, mean_pct=100 * err.mean(), med_pct=100 * np.median(err),
                    p95_pct=100 * np.percentile(err, 95), max_pct=100 * err.max())
    err = np.abs(y_pred - y_true)
    r2 = 1.0 - np.sum((y_pred - y_true) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return dict(r2=r2, mean_K=err.mean(), med_K=np.median(err), p95_K=np.percentile(err, 95), max_K=err.max())


def cv(Xn, y, degree, folds):
    pred = np.empty_like(y)
    for tr, va in folds:
        w, p = fit_tps(Xn[tr], y[tr], degree)
        pred[va] = pred_tps(Xn[va], Xn[tr], w, p, degree)
    return pred


def main():
    t_all = time.time()
    X, Q, T, cid = load()
    n = len(Q)
    print(f"[load] {CSV}: {n} 行, Q in [{Q.min():.9f}, {Q.max():.9f}] W/m, "
          f"tip_T in [{T.min():.6f}, {T.max():.6f}] K", flush=True)

    lo = np.array([BOUNDS[k][0] for k in LNAMES])
    hi = np.array([BOUNDS[k][1] for k in LNAMES])
    Xn = (X - lo) / (hi - lo)
    assert np.all(Xn >= 0) and np.all(Xn <= 1), "实测点越界，归一化失败"

    yQ = np.log(Q)
    yT = T.copy()

    perm = np.random.default_rng(SEED).permutation(n)
    folds = []
    for i in range(NFOLD):
        va = np.sort(perm[i::NFOLD])
        tr = np.sort(np.setdiff1d(perm, va, assume_unique=False))
        folds.append((tr, va))

    print(f"[cv] {NFOLD} 折, seed={SEED}", flush=True)
    rows_cv = []
    res = {}
    for deg in (0, 1):
        t0 = time.time()
        pQ = cv(Xn, yQ, deg, folds)
        pT = cv(Xn, yT, deg, folds)
        mQ = metrics(yQ, pQ, "lnQ")
        mT = metrics(yT, pT, "T")
        res[deg] = (mQ, mT)
        rows_cv.append(dict(degree=deg, target="Q", **mQ))
        rows_cv.append(dict(degree=deg, target="tip_T", **mT))
        print(f"[cv] degree={deg}  Q: R2(logQ)={mQ['r2_logQ']:.6f} R2(Q)={mQ['r2_Q']:.6f} "
              f"mean={mQ['mean_pct']:.3f}% med={mQ['med_pct']:.3f}% P95={mQ['p95_pct']:.3f}% max={mQ['max_pct']:.3f}%",
              flush=True)
        print(f"[cv] degree={deg}  tip_T: R2={mT['r2']:.6f} mean={mT['mean_K']:.4f}K "
              f"med={mT['med_K']:.4f}K P95={mT['p95_K']:.4f}K max={mT['max_K']:.4f}K  ({time.time()-t0:.1f}s)",
              flush=True)

    # 选阶：优先 degree=0（4096 权重 + 1 常数，正好 4097），若其 P95 超过 1 就换 degree=1
    deg = 0 if res[0][0]["p95_pct"] <= 1.0 * res[1][0]["p95_pct"] else 1
    print(f"[pick] degree={deg}  (P95: deg0={res[0][0]['p95_pct']:.3f}%  deg1={res[1][0]['p95_pct']:.3f}%)", flush=True)

    t0 = time.time()
    wQ, pQ = fit_tps(Xn, yQ, deg)
    wT, pT = fit_tps(Xn, yT, deg)
    print(f"[fit] full 4096 点求解完成 ({time.time()-t0:.1f}s), poly 项数={len(pQ)}", flush=True)

    # ---- 与 scipy 官方实现逐点对齐（同一份数据、同一 kernel/degree）
    t0 = time.time()
    scQ = RBFInterpolator(Xn, yQ, kernel="thin_plate_spline", smoothing=0.0, degree=deg)
    scT = RBFInterpolator(Xn, yT, kernel="thin_plate_spline", smoothing=0.0, degree=deg)
    dQ = np.abs(scQ(Xn) - pred_tps(Xn, Xn, wQ, pQ, deg)).max()
    dT = np.abs(scT(Xn) - pred_tps(Xn, Xn, wT, pT, deg)).max()
    print(f"[check] vs scipy RBFInterpolator: max|d lnQ|={dQ:.3e}  max|d tip_T|={dT:.3e}  ({time.time()-t0:.1f}s)", flush=True)

    # ---- 样本内残差（应接近 0，插值器性质）
    inQ = pred_tps(Xn, Xn, wQ, pQ, deg)
    inT = pred_tps(Xn, Xn, wT, pT, deg)
    in_err = np.abs(np.exp(inQ) - Q) / Q
    print(f"[check] in-sample Q 相对误差: max={100*in_err.max():.3e}%", flush=True)

    meta = dict(kernel="thin_plate_spline", degree=int(deg), smoothing=0.0, n_centers=int(n),
                inputs=LNAMES, units=dict(L_mm="mm", t_mm="mm", h="W/(m^2*K)", k="W/(m*K)"),
                norm="minmax", bounds={k: list(v) for k, v in BOUNDS.items()},
                targets=dict(Q="exp(lnQ) in W/m", tip_T="K"),
                cv=dict(seed=SEED, nfold=NFOLD,
                        Q_deg_selected=res[deg][0], T_deg_selected=res[deg][1],
                        Q_deg0=res[0][0], Q_deg1=res[1][0]),
                source_csv=CSV, build_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    npz = os.path.join(OUT, "surrogate.npz")
    np.savez_compressed(
        npz,
        centers=X.astype(np.float32),
        norm_min=lo, norm_max=hi,
        weights_Q=wQ.astype(np.float64), poly_Q=pQ.astype(np.float64),
        weights_T=wT.astype(np.float64), poly_T=pT.astype(np.float64),
        case_ids=cid.astype(np.int32),
        meta=json.dumps(meta, ensure_ascii=False),
    )
    print(f"[save] {npz}  {os.path.getsize(npz)} B  (weight 个数 {wQ.size}+{pQ.size} / {wT.size}+{pT.size})", flush=True)

    # ---- 10 折表
    with open(os.path.join(OUT, "cv_metrics.csv"), "w", newline="") as f:
        keys = ["degree", "target", "r2_logQ", "r2_Q", "mean_pct", "med_pct", "p95_pct", "max_pct",
                "r2", "mean_K", "med_K", "p95_K", "max_K"]
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows_cv:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    print("[save] cv_metrics.csv", flush=True)

    # ---- h=200,k=400 的 8x8 (L,t)->Q 表
    Ls = sorted(set(X[:, 0].tolist()))
    ts = sorted(set(X[:, 1].tolist()))
    sel = {}
    for i in range(n):
        if X[i, 2] == 200.0 and X[i, 3] == 400.0:
            sel[(X[i, 0], X[i, 1])] = (Q[i], T[i], cid[i])
    assert len(sel) == len(Ls) * len(ts), f"切片不完整: {len(sel)} != {len(Ls)*len(ts)}"
    with open(os.path.join(OUT, "table_Lt_Q.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["L_mm"] + [f"t={t:g}mm" for t in ts])
        for Lv in Ls:
            wr.writerow([f"{Lv:g}"] + [f"{sel[(Lv, tv)][0]:.6g}" for tv in ts])
    print(f"[save] table_Lt_Q.csv  ({len(Ls)}x{len(ts)} 全命中)", flush=True)

    print("=== 8x8 Q 表 (h=200,k=400) ===", flush=True)
    with open(os.path.join(OUT, "table_Lt_Q.csv")) as f:
        print(f.read(), flush=True)

    with open(os.path.join(OUT, "export_report.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(dict(meta=meta,
                                check_vs_scipy=dict(max_d_lnQ=float(dQ), max_d_tipT=float(dT)),
                                in_sample_max_rel_pct=float(100 * in_err.max()),
                                npz_bytes=os.path.getsize(npz)), ensure_ascii=False, indent=2))
    print(f"[done] 总耗时 {time.time()-t_all:.1f}s", flush=True)


if __name__ == "__main__":
    main()
