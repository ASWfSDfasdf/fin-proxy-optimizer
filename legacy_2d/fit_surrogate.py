#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_surrogate.py — 从扫描结果里拟合代理模型并给出最优参数（v0.1）

输入：fin2d.py --scan 产出的 results.csv
输出：代理模型精度（LOO）、最优参数点、以及可直接复算的验证命令

方法：径向基函数（RBF）插值 + 留一交叉验证（LOO）估计泛化误差，
      再在参数盒内做分块网格搜索取最优。纯 numpy，不依赖 sklearn。

为什么用 RBF 而不是神经网络：样本量通常只有几十到几百条，
RBF 在这种规模上更稳、无需调参、可解释（每个样本一个基）。

用法
  python fit_surrogate.py --csv scan_out/results.csv --target Q_W_per_m
  python fit_surrogate.py --csv ... --target tip_T_K --minimize
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

# 输出量/元数据不能当自变量，否则等于用答案预测答案
SKIP = {"case_id", "status", "Q_analytic", "Q_rel_err", "error_max",
        "error_mean", "theta_tip", "tip_T_K", "Q_W_per_m", "nx", "ny"}

# 只有这些才是「可设计参数」，常量（k/tb/tinf）不参与寻优
DESIGN = ("L", "t", "h", "k")


def _isnum(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def load(csv_path, target):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            try:
                y = float(r[target])
            except (KeyError, ValueError):
                continue
            rows.append((r, y))
    if not rows:
        raise SystemExit("读不到 %s 的数值行：%s" % (target, csv_path))

    keys = [k for k in rows[0][0]
            if k not in SKIP and k != target
            and all(_isnum(r.get(k)) for r, _ in rows)]
    X = np.array([[float(r[k]) for k in keys] for r, _ in rows])
    y = np.array([v for _, v in rows])
    return keys, X, y


def _pdist(Z):
    sq = (Z * Z).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


def _pdist_pair(A, B):
    sa = (A * A).sum(axis=1)[:, None]
    sb = (B * B).sum(axis=1)[None, :]
    d2 = sa + sb - 2.0 * A @ B.T
    np.maximum(d2, 0, out=d2)
    return np.sqrt(d2)


class RBF:
    """高斯 RBF 插值：f(x) = Σ w_i φ(‖x-x_i‖)，φ(r) = exp(-(r/σ)²)。

    σ 取样本间中位距离；对角线加极小正则保证可解。
    """

    def __init__(self, X, y, ridge=1e-10):
        self.xm = X.mean(axis=0)
        self.xs = X.std(axis=0)
        self.xs[self.xs == 0] = 1.0
        Z = (X - self.xm) / self.xs
        d = _pdist(Z)
        off = d[d > 0]
        self.sigma = float(np.median(off)) if off.size else 1.0
        self.Z = Z
        K = np.exp(-(d / self.sigma) ** 2) + ridge * np.eye(len(Z))
        self.w = np.linalg.solve(K, y)

    def predict(self, Xq):
        Zq = (Xq - self.xm) / self.xs
        d = _pdist_pair(Zq, self.Z)
        return np.exp(-(d / self.sigma) ** 2) @ self.w


def loo(X, y):
    """留一交叉验证：每次留一个样本，用其余样本拟合再预测它。"""
    preds = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        preds[i] = RBF(X[m], y[m]).predict(X[i:i + 1])[0]
    err = preds - y
    rmse = float(np.sqrt((err ** 2).mean()))
    r2 = float(1.0 - (err ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(y), 1e-12))) * 100)
    return rmse, r2, mape


def search_box(model, X, keys, grid_hint):
    """在「可设计且确实在变」的维度上做分块网格搜索，返回 (最优值, 最优点 dict)。"""
    varying = [i for i in range(X.shape[1]) if X[:, i].std() > 0]
    dims = [i for i in varying if keys[i] in DESIGN] or varying
    if not dims:
        raise SystemExit("没有可变的参数维度，无法寻优")

    lo, hi = X.min(axis=0), X.max(axis=0)
    budget = 500_000
    per = max(6, min(grid_hint, int(round(budget ** (1.0 / len(dims))))))
    axes = [np.linspace(lo[i], hi[i], per) for i in dims]
    total = per ** len(dims)
    print("")
    print("搜索 %d 维（%s），每维 %d 点，共 %d 个候选"
          % (len(dims), ", ".join(keys[i] for i in dims), per, total))

    mesh = np.meshgrid(*axes, indexing="ij")
    Qd = np.stack([m.ravel() for m in mesh], axis=1)

    base_vec = X.mean(axis=0)
    best_val, best_vec = None, None
    step = 100_000
    for s0 in range(0, len(Qd), step):
        blk = np.tile(base_vec, (len(Qd[s0:s0 + step]), 1))
        blk[:, dims] = Qd[s0:s0 + step]
        pv = model.predict(blk)
        kk = int(np.argmin(pv) if MINIMIZE else np.argmax(pv))
        v = float(pv[kk])
        better = (best_val is None) or (v < best_val if MINIMIZE else v > best_val)
        if better:
            best_val, best_vec = v, blk[kk].copy()

    best = {keys[i]: float(best_vec[i]) for i in dims}
    return float(best_val), best


MINIMIZE = False


def main(argv=None):
    global MINIMIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--target", default="Q_W_per_m")
    ap.add_argument("--minimize", action="store_true", help="默认取最大值")
    ap.add_argument("--grid", type=int, default=25, help="每维搜索格点数（上限）")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    MINIMIZE = bool(args.minimize)

    keys, X, y = load(args.csv, args.target)
    print("样本 %d 条，自变量 %s" % (len(y), keys))
    print("目标 %s：min=%.4f max=%.4f" % (args.target, y.min(), y.max()))

    rmse, r2, mape = loo(X, y)
    print("")
    print("LOO：RMSE=%.4g  R2=%.4f  MAPE=%.2f%%" % (rmse, r2, mape))
    if r2 < 0.9:
        print("[warn] R2 < 0.9：样本覆盖不足或目标噪声大，结论只当参考")

    model = RBF(X, y)
    pred_best, best = search_box(model, X, keys, args.grid)

    print("")
    print("代理模型最优（%s %s）：%.4f"
          % ("最小" if MINIMIZE else "最大", args.target, pred_best))
    for kk, vv in best.items():
        print("  %s = %.6g" % (kk, vv))

    case = {kk: round(vv, 6) for kk, vv in best.items()}
    print("")
    print("下一步：把最优点丢回真求解器复核")
    print("  1) 建文件 best.json，内容：%s" % json.dumps(case, ensure_ascii=False))
    print("  2) python fin2d.py --case best.json --out best_profile.csv")

    if args.json_out:
        json.dump({"target": args.target, "loo_rmse": rmse, "loo_r2": r2,
                   "loo_mape": mape, "predicted": pred_best,
                   "best": best, "n": int(len(y))},
                  open(args.json_out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
