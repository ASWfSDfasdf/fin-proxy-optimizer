#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict.py —— 3D 单肋稳态导热 RBF 代理模型的独立推理器。

只依赖 numpy（不依赖 scipy）：thin-plate-spline 的 phi(r)=r^2*ln r 与矩阵乘全部自带实现。

用法
----
    python predict.py 50 2 40 200          # 单点：L=50mm, t=2mm, h=40, k=200
    python predict.py --bench              # 计时（1 万次调用）
    python predict.py --interp-check results.csv   # 插值自洽性检查（不是泛化精度）

Python 内
---------
    from predict import Surrogate
    m = Surrogate("surrogate.npz")
    r = m.predict(50.0, 2.0, 40.0, 200.0)
    # r = {"status": "ok", "Q_W_per_m": ..., "tip_T_K": ...}
    # 越界: {"status": "超范围", "reason": "L=120 超出 [30, 100]", "Q_W_per_m": None, "tip_T_K": None}

边界
----
    仅在内 [30,100]mm x [1,5]mm x [10,200] x [50,400] 内有效；越界一律返回"超范围"，
    绝不外推（RBF 在凸包外没有物理意义，且 TPS 在远处发散）。
"""
import sys
import time
import numpy as np

BOUNDS_ORDER = ("L_mm", "t_mm", "h", "k")
DEFAULT_BOUNDS = ((30.0, 100.0), (1.0, 5.0), (10.0, 200.0), (50.0, 400.0))
OUT_STATUS = "超范围"


def _phi_from_r2(r2):
    """phi(r) = r^2 ln r = 0.5 * r2 * ln(r2)；r=0 处取 0（TPS 的极限值）。"""
    out = np.zeros_like(r2)
    m = r2 > 0.0
    rr = r2[m]
    out[m] = 0.5 * rr * np.log(rr)
    return out


class Surrogate:
    def __init__(self, path="surrogate.npz"):
        d = np.load(path, allow_pickle=False)
        self.centers = np.asarray(d["centers"], dtype=np.float64)     # (n,4) 物理量
        self.lo = np.asarray(d["norm_min"], dtype=np.float64)         # (4,)
        self.hi = np.asarray(d["norm_max"], dtype=np.float64)
        self.wQ = np.asarray(d["weights_Q"], dtype=np.float64)        # (n,)
        self.pQ = np.asarray(d["poly_Q"], dtype=np.float64)           # (m,)
        self.wT = np.asarray(d["weights_T"], dtype=np.float64)
        self.pT = np.asarray(d["poly_T"], dtype=np.float64)
        if "meta" in d.files:
            import json
            try:
                self.meta = json.loads(str(d["meta"]))
            except Exception:
                self.meta = {}
        else:
            self.meta = {}
        self.degree = int(self.meta.get("degree", len(self.pQ) - 1))
        self.centers_n = (self.centers - self.lo) / (self.hi - self.lo)
        self.bounds = tuple(tuple(self.meta.get("bounds", {}).get(k, DEFAULT_BOUNDS[i]))
                            for i, k in enumerate(BOUNDS_ORDER))

    # ---------- 内部 ----------
    def _poly(self, Xn):
        if self.degree == 0:
            return np.ones((Xn.shape[0], 1))
        return np.hstack([np.ones((Xn.shape[0], 1)), Xn])

    def _phi(self, Xn, chunk=1024):
        n = Xn.shape[0]
        out = np.empty((n, self.centers_n.shape[0]))
        for i in range(0, n, chunk):
            d = Xn[i:i + chunk, None, :] - self.centers_n[None, :, :]
            out[i:i + chunk] = _phi_from_r2(np.einsum("ijk,ijk->ij", d, d))
        return out

    def _check(self, x):
        for i, name in enumerate(BOUNDS_ORDER):
            lo, hi = self.bounds[i]
            if not np.isfinite(x[i]):
                return f"{name}={x[i]} 不是有限数"
            if x[i] < lo or x[i] > hi:
                return f"{name}={x[i]:g} 超出 [{lo:g}, {hi:g}]"
        return None

    # ---------- 对外 ----------
    def predict(self, L, t, h, k):
        """单点预测。返回 dict；越界时 status='超范围'，Q/tip_T 为 None（不写 nan）。"""
        x = np.array([L, t, h, k], dtype=np.float64)
        bad = self._check(x)
        if bad is not None:
            return {"status": OUT_STATUS, "reason": bad, "Q_W_per_m": None, "tip_T_K": None}
        return self._predict_checked(x.reshape(1, 4))[0]

    def predict_many(self, X):
        """批量预测，X 为 (m,4)。越界行单独标 '超范围'，其余照算（不整批失败）。"""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        out = [None] * X.shape[0]
        good, idx = [], []
        for i in range(X.shape[0]):
            bad = self._check(X[i])
            if bad is None:
                good.append(i)
                idx.append(i)
            else:
                out[i] = {"status": OUT_STATUS, "reason": bad, "Q_W_per_m": None, "tip_T_K": None}
        if good:
            res = self._predict_checked(X[good])
            for j, i in enumerate(good):
                out[i] = res[j]
        return out

    def _predict_checked(self, X):
        Xn = (X - self.lo) / (self.hi - self.lo)
        K = self._phi(Xn)
        P = self._poly(Xn)
        lnQ = K @ self.wQ + P @ self.pQ
        T = K @ self.wT + P @ self.pT
        Q = np.exp(lnQ)
        return [{"status": "ok", "Q_W_per_m": float(Q[i]), "tip_T_K": float(T[i]),
                 "lnQ": float(lnQ[i])} for i in range(X.shape[0])]


# ---------------- CLI ----------------
def _fmt(v):
    return "None" if v is None else f"{v:.6f}"


def main(argv):
    if len(argv) > 1 and argv[1] == "--bench":
        m = Surrogate()
        x = np.array([50.0, 2.0, 40.0, 200.0])
        m.predict(*x)
        n = 10000
        t0 = time.perf_counter()
        for _ in range(n):
            m.predict(*x)
        dt = time.perf_counter() - t0
        print(f"bench: {n} 次单点调用 {dt*1000:.1f} ms  ->  {dt/n*1e6:.1f} us/次")
        Xb = np.tile(np.array([50.0, 2.0, 40.0, 200.0]), (1000, 1))
        t0 = time.perf_counter()
        m.predict_many(Xb)
        print(f"bench: 1000 点批量 {(time.perf_counter()-t0)*1000:.1f} ms")
        return 0

    if len(argv) > 1 and argv[1] in ("--interp-check", "--selftest"):
        if argv[1] == "--selftest":
            print("[注意] --selftest 已更名为 --interp-check，本次按新名字执行")
        # 这两个数字相差 10 个数量级，必须先把话说在前头，否则必被误读
        print("插值自洽性检查，非泛化精度")
        print("  泛化精度见 10 折 CV：P95 = 4.318%（Q） / 0.369 K（tip_T），平均 1.359%")
        import csv
        rows = list(csv.DictReader(open(argv[2], newline="")))
        m = Surrogate()
        X = np.array([[float(r["L_mm"]), float(r["t_mm"]), float(r["h"]), float(r["k"])] for r in rows])
        Q = np.array([float(r["Q_W_per_m"]) for r in rows])
        T = np.array([float(r["tip_T_K"]) for r in rows])
        res = m.predict_many(X)
        bad = sum(1 for r in res if r["status"] != "ok")
        qp = np.array([r["Q_W_per_m"] for r in res if r["status"] == "ok"])
        tp = np.array([r["tip_T_K"] for r in res if r["status"] == "ok"])
        e = np.abs(qp - Q) / Q
        et = np.abs(tp - T)
        print(f"interp-check: {len(rows)} 点（全部为训练点）, 越界 {bad} 点")
        print(f"  Q  相对误差: 平均 {100*e.mean():.3e}%  P95 {100*np.percentile(e,95):.3e}%  最大 {100*e.max():.3e}%")
        print(f"  tip_T 绝差 : 平均 {et.mean():.3e} K  P95 {np.percentile(et,95):.3e} K  最大 {et.max():.3e} K")
        print("  以上≈0 是插值器的本征结果，不代表对未见参数点的预测能力")
        return 0 if bad == 0 else 1

    if len(argv) != 5:
        print(__doc__)
        return 2
    L, t, h, k = (float(v) for v in argv[1:5])
    r = Surrogate().predict(L, t, h, k)
    if r["status"] != "ok":
        print(f"{OUT_STATUS}：{r['reason']}")
        return 1
    print(f"L={L:g} mm  t={t:g} mm  h={h:g} W/(m^2K)  k={k:g} W/(mK)")
    print(f"Q     = {_fmt(r['Q_W_per_m'])} W/m")
    print(f"tip_T = {_fmt(r['tip_T_K'])} K")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
