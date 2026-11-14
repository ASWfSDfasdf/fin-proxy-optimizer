#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fin2d.py — 二维单肋稳态导热的数值解（v0.1）

物理：一块矩形肋片（长 L、厚 t、导热系数 k），根部温度 tb，
两侧与端部以对流换热系数 h 与温度 tinf 的环境换热，
沿肋片方向一维、厚度方向二维，求解稳态温度场 T(x,y)。

边界条件
  x = 0（根部）   : T = tb                      （第一类）
  x = L（端部）   : -k dT/dx = h (T - tinf)      （第三类）
  y = 0, y = t     : ±k dT/dy = h (T - tinf)     （第三类）

数值方法：均匀网格上的五点差分，scipy.sparse 直接求解。
单位：长度 m，温度 K，导热系数 W/(m·K)，换热系数 W/(m²·K)。
热流 Q 的单位是 W/m（沿展向单位深度，即每 1 m 宽）。

用法
  python fin2d.py --check                       # 与一维解析解对比（自检）
  python fin2d.py --case case.json              # 单算例，写 CSV
  python fin2d.py --scan scan.json --outdir out # 批量扫描，汇总 results.csv

依赖：numpy、scipy（集群上 module load anaconda3/2023.09 即有）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

try:                                    # scipy 有就用稀疏直接解（集群上都有）
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve
    HAVE_SCIPY = True
except ImportError:                     # 本地没有 scipy 时退回 numpy 稠密解（仅小网格）
    HAVE_SCIPY = False
DENSE_LIMIT = 6000                      # 稠密解的未知量上限

DEFAULTS = {
    "L": 0.05,      # 肋高 m
    "t": 0.002,     # 肋厚 m
    "k": 200.0,     # 导热系数 W/(m·K)
    "h": 50.0,      # 对流换热系数 W/(m²·K)
    "tb": 350.0,    # 根部温度 K
    "tinf": 300.0,  # 环境温度 K
    "nx": 201,      # 长度方向节点数（>=3）
    "ny": 41,       # 厚度方向节点数（>=3）
}


# ---------------------------------------------------------------- 解析解
def analytic(case: dict, xs):
    """一维肋片解析解（把厚度方向的侧面对流折算成 m）。

    这是二维解在 Bi = h·t/k → 0 时的极限，用来给数值解做对照。
      m² = 2h / (k·t)        两侧对流 / 截面积
      θ(x)/θb = [cosh(m(L-x)) + β sinh(m(L-x))] / [cosh(mL) + β sinh(mL)]
      β = h / (m·k)          端部对流
    返回 (theta_over_thetab, Q_per_depth)
    """
    L, t, k, h = case["L"], case["t"], case["k"], case["h"]
    tb, tinf = case["tb"], case["tinf"]
    theta_b = tb - tinf
    m = math.sqrt(2.0 * h / (k * t))
    beta = h / (m * k)
    mL = m * L
    den = math.cosh(mL) + beta * math.sinh(mL)

    theta = []
    for x in xs:
        mLx = m * (L - x)
        theta.append((math.cosh(mLx) + beta * math.sinh(mLx)) / den)

    # Q = θb · sqrt(h·P·k·A)，P=2（两侧），A=t（单位展向）
    Q = theta_b * math.sqrt(h * 2.0 * k * t) * \
        (math.sinh(mL) + beta * math.cosh(mL)) / den
    return np.array(theta), Q


# ---------------------------------------------------------------- 数值解
def solve(case: dict):
    """求解二维温度场，返回 dict（含 T 矩阵、根部热流、与解析解的误差）。"""
    c = dict(DEFAULTS)
    c.update({k: v for k, v in case.items() if v is not None})
    L, t, k, h = float(c["L"]), float(c["t"]), float(c["k"]), float(c["h"])
    tb, tinf = float(c["tb"]), float(c["tinf"])
    nx, ny = int(c["nx"]), int(c["ny"])
    if nx < 3 or ny < 3:
        raise ValueError("nx/ny 至少为 3")

    dx = L / (nx - 1)
    dy = t / (ny - 1)
    n = nx * ny

    rows, cols, vals, rhs = [], [], [], np.zeros(n)
    idx = lambda i, j: i * ny + j

    kdx2, kdy2 = k / dx**2, k / dy**2
    kdx, kdy = k / dx, k / dy
    alpha = k * dy / (2.0 * dx**2)      # 上下表面半格的 x 向传导系数
    beta = k * dx / (2.0 * dy**2)       # 端部半格的 y 向传导系数

    for i in range(nx):
        for j in range(ny):
            p = idx(i, j)

            if i == 0:                                   # 根部：定温
                rows.append(p); cols.append(p); vals.append(1.0)
                rhs[p] = tb
                continue

            if i == nx - 1:                              # 端部半格：x 向对流 + y 向传导
                diag = kdx + h
                off = [(idx(i - 1, j), -kdx)]
                if 0 < j < ny - 1:
                    diag += 2.0 * beta
                    off += [(idx(i, j + 1), -beta), (idx(i, j - 1), -beta)]
                rows.append(p); cols.append(p); vals.append(diag)
                for cc, vv in off:
                    rows.append(p); cols.append(cc); vals.append(vv)
                rhs[p] = h * tinf
                continue

            if j == 0 or j == ny - 1:                    # 上下表面半格：y 向对流 + x 向传导
                jn = 1 if j == 0 else ny - 2
                diag = kdy + h + 2.0 * alpha
                off = [(idx(i, jn), -kdy),
                       (idx(i + 1, j), -alpha),
                       (idx(i - 1, j), -alpha)]
                rows.append(p); cols.append(p); vals.append(diag)
                for cc, vv in off:
                    rows.append(p); cols.append(cc); vals.append(vv)
                rhs[p] = h * tinf
                continue

            rows += [p, p, p, p, p]                      # 内部：五点差分
            cols += [p, idx(i + 1, j), idx(i - 1, j), idx(i, j + 1), idx(i, j - 1)]
            vals += [-2 * kdx2 - 2 * kdy2, kdx2, kdx2, kdy2, kdy2]

    if HAVE_SCIPY:
        A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        T = spsolve(A, rhs).reshape(nx, ny)
    elif n <= DENSE_LIMIT:
        A = np.zeros((n, n))
        A[rows, cols] = vals
        T = np.linalg.solve(A, rhs).reshape(nx, ny)
    else:
        raise RuntimeError(
            "当前环境没有 scipy，且未知量 %d > %d 超出稠密解上限。"
            "请在集群上 module load anaconda3/2023.09 后运行，或 pip install scipy。"
            % (n, DENSE_LIMIT))

    # 根部热流（W/m 展向）：用二阶单侧差分，避免 O(dx) 误差主导
    #   dT/dx|_0 ≈ (-3T0 + 4T1 - T2) / (2dx)
    dTdx0 = (-3.0 * T[0, :] + 4.0 * T[1, :] - T[2, :]) / (2.0 * dx)
    # 沿厚度用梯形积分（两端各半格），矩形积分会多算一个 dy
    w = np.full(ny, dy)
    w[0] = w[-1] = 0.5 * dy
    Q = float(-k * np.sum(dTdx0 * w))
    # 端部温差（用于看端部是否真的被对流拉低）
    tip_T = float(np.mean(T[-1, :]))

    xs = np.linspace(0.0, L, nx)
    theta_an, Q_an = analytic({**c, "L": L, "t": t, "k": k, "h": h,
                               "tb": tb, "tinf": tinf}, xs)
    T_mid = T[:, ny // 2]
    theta_num = (T_mid - tinf) / max(tb - tinf, 1e-12)

    # 误差只在 i>=1 上统计（根部是强加的定温）
    rel = np.abs(theta_num[1:] - theta_an[1:]) / np.maximum(np.abs(theta_an[1:]), 1e-12)

    return {
        "case": {**c, "L": L, "t": t, "k": k, "h": h, "tb": tb, "tinf": tinf,
                 "nx": nx, "ny": ny},
        "T": T,
        "x": xs,
        "theta_num": theta_num,
        "theta_analytic": theta_an,
        "Q": Q,
        "Q_analytic": Q_an,
        "Q_rel_err": abs(Q - Q_an) / max(abs(Q_an), 1e-12),
        "tip_T": tip_T,
        "theta_tip": float(theta_num[-1]),
        "error_max": float(rel.max()),
        "error_mean": float(rel.mean()),
    }


# ---------------------------------------------------------------- 输出
def write_profile_csv(res, path):
    c = res["case"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x_m", "T_mean_K", "theta_num", "theta_analytic", "rel_err"])
        for i, x in enumerate(res["x"]):
            an = res["theta_analytic"][i]
            num = res["theta_num"][i]
            err = abs(num - an) / max(abs(an), 1e-12)
            w.writerow([f"{x:.6g}", f"{res['theta_num'][i] * (c['tb'] - c['tinf']) + c['tinf']:.4f}",
                        f"{num:.6g}", f"{an:.6g}", f"{err:.6g}"])


def summary_row(cid, res, extra=None):
    c = res["case"]
    row = {
        "case_id": cid, "L": c["L"], "t": c["t"], "k": c["k"], "h": c["h"],
        "tb": c["tb"], "tinf": c["tinf"], "nx": c["nx"], "ny": c["ny"],
        "Q_W_per_m": round(res["Q"], 6), "Q_analytic": round(res["Q_analytic"], 6),
        "Q_rel_err": round(res["Q_rel_err"], 6),
        "tip_T_K": round(res["tip_T"], 4),
        "Q_per_vol": round(res["Q"] / max(c["L"] * c["t"], 1e-12), 3),
        "theta_tip": round(res["theta_tip"], 6),
        "error_max": round(res["error_max"], 6),
        "error_mean": round(res["error_mean"], 6),
    }
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------- 自检
def run_check(verbose=True):
    """自检：把二维数值解与一维解析解对比，并做网格加密收敛检查。"""
    case = {**DEFAULTS, "nx": 401 if HAVE_SCIPY else 79}
    ny_list = (11, 21, 41, 81) if HAVE_SCIPY else (11, 17, 25, 33)
    ok = True
    print("=" * 68)
    print("自检 1/2：二维解 vs 一维解析解（Bi = h·t/k = %.2e）"
          % (case["h"] * case["t"] / case["k"]))
    print("=" * 68)
    print(f"{'ny':>5} {'Q_numeric':>12} {'Q_analytic':>12} {'Q_err':>9} "
          f"{'T_err_max':>11} {'T_err_mean':>11}")
    prev = None
    for ny in ny_list:
        r = solve({**case, "ny": ny})
        print(f"{ny:5d} {r['Q']:12.4f} {r['Q_analytic']:12.4f} "
              f"{r['Q_rel_err']*100:8.3f}% {r['error_max']*100:10.3f}% "
              f"{r['error_mean']*100:10.3f}%")
        if prev is not None:
            # 网格加密后误差应下降（允许最后一级因浮点/离散极限持平）
            if r["error_max"] > prev * 1.05:
                ok = False
                print(f"  [FAIL] ny={ny} 误差未随网格加密下降")
        prev = r["error_max"]

    base = solve({**case, "ny": 41})
    print()
    print("自检 2/2：能量与边界一致性")
    print(f"  根部热流 Q = {base['Q']:.4f} W/m（解析 {base['Q_analytic']:.4f}，"
          f"误差 {base['Q_rel_err']*100:.3f}%）")
    print(f"  端部温度   = {base['tip_T']:.3f} K（θ/θb = {base['theta_tip']:.4f}）")
    checks = [
        ("阈值：温度场最大相对误差 < 1%", base["error_max"] < 0.01),
        ("阈值：根部热流相对误差 < 2%", base["Q_rel_err"] < 0.02),
        ("物理：端部温度低于根部", base["tip_T"] < base["case"]["tb"]),
        ("物理：端部温度高于环境温度", base["tip_T"] > base["case"]["tinf"]),
    ]
    for name, good in checks:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
        ok = ok and good
    print()
    print("自检结果：", "OK" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="二维单肋稳态导热求解器")
    ap.add_argument("--check", action="store_true", help="与解析解对比自检")
    ap.add_argument("--case", help="单算例 JSON（参数名见 DEFAULTS）")
    ap.add_argument("--out", default="profile.csv", help="单算例温度分布输出 CSV")
    ap.add_argument("--summary", default=None, help="单算例汇总 JSON 输出")
    ap.add_argument("--scan", help="批量扫描 JSON：{base:{...}, grid:{L:[..], t:[..]}}")
    ap.add_argument("--outdir", default="scan_out", help="扫描输出目录")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.check:
        return 0 if run_check() else 1

    if args.case:
        case = json.load(open(args.case, encoding="utf-8")) if os.path.exists(args.case) else {}
        res = solve(case)
        write_profile_csv(res, args.out)
        row = summary_row("single", res)
        if args.summary:
            json.dump(row, open(args.summary, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        print(json.dumps(row, ensure_ascii=False))
        return 0

    if args.scan:
        spec = json.load(open(args.scan, encoding="utf-8"))
        base = spec.get("base", {})
        grid = spec.get("grid", {})
        keys = list(grid.keys())
        combos = [{}]
        for kkey in keys:
            combos = [dict(c, **{kkey: v}) for c in combos for v in grid[kkey]]

        os.makedirs(args.outdir, exist_ok=True)
        rows = []
        for i, over in enumerate(combos, 1):
            case = {**DEFAULTS, **base, **over}
            try:
                res = solve(case)
            except Exception as e:                       # 单个失败不影响整批
                rows.append({"case_id": i, "status": f"error: {e}", **over})
                continue
            cid = f"{i:05d}"
            write_profile_csv(res, os.path.join(args.outdir, f"profile_{cid}.csv"))
            rows.append(summary_row(cid, res, extra={"status": "ok", **over}))
            if not args.quiet:
                print(f"[{i}/{len(combos)}] Q={res['Q']:.4f} W/m "
                      f"T_err={res['error_max']*100:.3f}% {over}")

        out = os.path.join(args.outdir, "results.csv")
        cols = sorted({k for r in rows for k in r})
        first = ["case_id", "status", "Q_W_per_m", "Q_rel_err", "error_max", "tip_T_K"]
        cols = first + [c for c in cols if c not in first]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        ok = sum(1 for r in rows if r.get("status") == "ok")
        print(f"\n完成：{ok}/{len(rows)} 个算例成功 → {out}")
        return 0 if ok else 2

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
