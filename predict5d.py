#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict5d.py — 5D single-fin surrogate forward predictor (pure numpy).

Inputs (mm / SI) : L_mm, t_mm, W_mm, h [W/(m^2 K)], k [W/(m K)]
Outputs          : Q_W_per_m, tip_L_T_K, T_max_K

The historical column tip_T_K (span end-face mean, not the tip) is NOT modelled.

Model = thin-plate-spline RBF with 32,768 centres, phi(r) = r, degree 0 (+const),
ridge 1e-10 on the kernel diagonal, inputs min-max normalised to [0,1]^5.

Usage
  python predict5d.py --selftest
  python predict5d.py --case 60,2.0,40,200,200 --case 100,1.0,160,10,400
  python predict5d.py --csv in.csv --out out.csv
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

DEFAULT_MODEL = "surrogate5d.npz"
COLS = ["L_mm", "t_mm", "W_mm", "h", "k"]


class Surrogate5D:
    def __init__(self, path: str = DEFAULT_MODEL):
        d = np.load(path, allow_pickle=False)
        self.centers = np.asarray(d["centers_mm"], dtype=np.float64)
        self.coef = np.asarray(d["coef"], dtype=np.float64)
        self.lo = np.asarray(d["lo"], dtype=np.float64)
        self.span = np.asarray(d["span"], dtype=np.float64)
        self.targets = [str(s) for s in d["targets"]]
        self.n_centers = int(self.centers.shape[0])
        self.hi = self.lo + self.span
        self._cs = (self.centers - self.lo) / self.span

    def predict(self, X, block: int = 512):
        """X: (n,5) physical units. Returns (Y, ok_mask)."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        tol = 1e-9 * np.maximum(1.0, np.abs(self.hi))
        ok = np.all((X >= self.lo - tol) & (X <= self.hi + tol), axis=1)
        Y = np.full((X.shape[0], len(self.targets)), np.nan)
        if not np.any(ok):
            return Y, ok
        Xn = (X[ok] - self.lo) / self.span
        out = np.empty((Xn.shape[0], len(self.targets)))
        cs2 = (self._cs * self._cs).sum(axis=1)          # (m,)
        for i in range(0, Xn.shape[0], block):
            q = Xn[i:i + block]
            d2 = (q * q).sum(axis=1)[:, None] + cs2[None, :] - 2.0 * (q @ self._cs.T)
            np.maximum(d2, 0.0, out=d2)
            K = np.sqrt(d2, out=d2)
            out[i:i + block] = K @ self.coef[:self.n_centers, :] + self.coef[self.n_centers, :]
        Y[ok] = out
        return Y, ok


def _fmt(v):
    return "%.6g" % v


def main(argv=None):
    ap = argparse.ArgumentParser(description="5D fin surrogate predictor")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--case", action="append", default=[],
                    help="comma separated L_mm,t_mm,W_mm,h,k (repeatable)")
    ap.add_argument("--csv", help="input csv with header L_mm,t_mm,W_mm,h,k")
    ap.add_argument("--out", help="output csv path (only with --csv)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    m = Surrogate5D(args.model)
    print("model   : %s" % args.model)
    print("centres : %d ; targets: %s" % (m.n_centers, ", ".join(m.targets)))
    print("bounds  : " + " ".join("%s[%g,%g]" % (c, m.lo[i], m.hi[i])
                                  for i, c in enumerate(COLS)))

    if args.selftest:
        d = np.load(args.model, allow_pickle=False)
        Xs = np.asarray(d["selftest_X"], dtype=np.float64)
        ys = np.asarray(d["selftest_y"], dtype=np.float64)
        Y, ok = m.predict(Xs)
        if not np.all(ok):
            print("WARN: %d selftest points out of range" % int((~ok).sum()))
        rel = np.abs(Y - ys) / np.maximum(np.abs(ys), 1e-30)
        print("")
        print("--selftest  n=%d training points (in-sample)" % Xs.shape[0])
        print("  注: 这是插值自洽性, 非泛化精度; 泛化精度见 LOLO 与 10% 留出.")
        for j, t in enumerate(m.targets):
            col = rel[ok, j]
            print("  %-12s rel |err|  median=%.3e  P95=%.3e  max=%.3e"
                  % (t, np.median(col), np.percentile(col, 95), col.max()))
        return 0

    rows = []
    if args.case:
        for s in args.case:
            parts = [p for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) != 5:
                print("bad --case (need 5 values): %s" % s, file=sys.stderr)
                return 2
            rows.append([float(p) for p in parts])
    elif args.csv:
        with open(args.csv, newline="") as f:
            for r in csv.DictReader(f):
                rows.append([float(r[c]) for c in COLS])
    else:
        ap.print_help()
        return 2

    X = np.asarray(rows, dtype=np.float64)
    Y, ok = m.predict(X)
    print("")
    print("%8s %8s %8s %8s %8s | %14s %12s %12s  %s"
          % ("L_mm", "t_mm", "W_mm", "h", "k", "Q_W_per_m", "tip_L_T_K", "T_max_K", "status"))
    for i in range(X.shape[0]):
        if not ok[i]:
            print("%8g %8g %8g %8g %8g | %14s %12s %12s  超范围"
                  % (*X[i], "-", "-", "-"))
        else:
            print("%8g %8g %8g %8g %8g | %14.6f %12.4f %12.4f  ok"
                  % (*X[i], Y[i, 0], Y[i, 1], Y[i, 2]))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS + m.targets + ["status"])
            for i in range(X.shape[0]):
                if ok[i]:
                    w.writerow([_fmt(v) for v in X[i]] + [round(float(v), 6) for v in Y[i]] + ["ok"])
                else:
                    w.writerow([_fmt(v) for v in X[i]] + ["", "", ""] + ["超范围"])
        print("\nwrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
