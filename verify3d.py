#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify3d.py —— fin3d 验收 (第三轮: A-1 终版 + R-2 + R-3 + E-1/E-2/E-3).

用法:  python verify3d.py [e1|e2|e3|all]      默认 e1

已冻结判据
  E-1(1)  |Q/W - 解析(端面对流)| / 解析           < 1e-3
  E-1(2a) 矩阵反推 ΣA_d vs A_total                < 1e-12   (纯几何, 不解方程)
  E-1(2b) 常温度场逐行残差 vs 预期面通量           < 1e-12   (相对 max|b|)
  E-1(2c) k=1e5, |Q/Qexp-1|                       < 1e-3    (预期 -mL^2/3)
  A-1     |Q_in-Q_out| <= max(1e-9*Q, 1e-6 W)     (终版, 替换 R-1)
          同时必须报: 绝对差(W)、相对差、当次 Q 值(6 位小数)
  E-2     三档 r=2 三向同步减半; 每档七项; p in [1.90,2.10]; GCI_fine < 0.5%
  E-3     W=25t/50t/100t, Δz 固定, Q/W 对 1/W 线性外推; 截距 vs 210.888 < 0.5%

已删除: k=1e8 等温极限 (测的是 float64 分辨率, 不是求解器)
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

import fin3d as F

try:                                     # 让远端日志即使中途被杀也可读
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

T0 = time.perf_counter()
FAILS = []
Q2D_REF = 210.888
CG_RTOL = 1.0e-12

BASE = dict(L=0.05, t=0.002, W=0.02, k=200.0, h=50.0, Tb=350.0, Tinf=300.0)


def check(tag, ok, text):
    print("%s  %-40s %s" % ("[PASS]" if ok else "[FAIL]", tag, text))
    if not ok:
        FAILS.append(tag)
    return ok


def hr(title):
    print("\n" + "=" * 92)
    print(title)
    print("=" * 92)


def a1(tag, r):
    """A-1 终版: |Q_in-Q_out| <= max(1e-9*Q, 1e-6 W)."""
    q = abs(r["Q"])
    ad = r["abs_diff"]
    rel = abs(r["imbalance"])
    tol = max(1e-9 * q, 1e-6)
    print("  A-1: |Qin-Qout| = %.6e W   |dQ|/Q = %.6e   Q = %.6f W"
          % (ad, rel, q))
    print("       阈值 max(1e-9*Q, 1e-6 W) = %.6e W" % tol)
    return check(tag, ad <= tol, "(%.3e / %.3e)" % (ad, tol))


# ---------------------------------------------------------------- E-1
def gate_e1():
    hr("E-1 (1)  一维退化: Ny=Nz=1, z 面绝热, Nx=2000  vs 端面对流解析解")
    p1 = dict(BASE, z_bc="adiabatic")
    q_ref = F.analytic_fin(p1, with_tip=True)
    q_adiab = F.analytic_fin(p1, with_tip=False)
    r = F.solve(p1, 2000, 1, 1, solver="direct")
    err = (r["Q_per_span"] - q_ref) / q_ref
    print("  解析 Q (端面对流, beta=h/mk) = %.6f W/m" % q_ref)
    print("  解析 Q (绝热端, 口径参照)   = %.6f W/m" % q_adiab)
    print("  三维码 Q/W                  = %.6f W/m" % r["Q_per_span"])
    print("  相对误差                    = %+.6e" % err)
    print("  能量: |Qin-Qout| = %.6e W   |dQ|/Q = %.6e"
          % (r["abs_diff"], abs(r["imbalance"])))
    print("  solver=%s iters=%d wall=%.3f s resid=%.3e"
          % (r["solve"]["method"], r["solve"]["iters"], r["solve"]["wall_s"],
             r["solve"]["residual"]))
    ok1 = check("E-1(1) 一维退化 <0.1%", abs(err) < 1e-3, "(%+.4f%%)" % (100 * err))
    oka = a1("A-1 能量(1D)", r)
    return ok1, oka


def r2a():
    hr("R-2 (a) 面积自洽: 从矩阵反推 ΣA_d, 与 A_total=2LW+2Lt+tW 比")
    asm = F.assemble(BASE, 80, 12, 32)
    sum_b = float(np.sum(asm["diag"] - asm["offsum"]))      # Rx*n_root + Σh_d
    n_root = float(np.sum(asm["c_root"]))
    sum_hd = sum_b - asm["Rx"] * n_root
    fac = lambda a, d: (1.0 / asm["h"] + d / (2.0 * asm["kc"]))
    a_re = (asm["Hx"] * float(np.sum(asm["c_tip"])) * fac(asm["Ax"], asm["dx"])
            + asm["Hy"] * float(np.sum(asm["c_y"])) * fac(asm["Ay"], asm["dy"])
            + asm["Hz"] * float(np.sum(asm["c_z"])) * fac(asm["Az"], asm["dz"]))
    a_tot = 2.0 * asm["L"] * asm["W"] + 2.0 * asm["L"] * asm["t"] + asm["t"] * asm["W"]
    rel = abs(a_re - a_tot) / a_tot
    print("  面计数: 端面 %d, y 面 %d, z 面 %d" % (int(np.sum(asm["c_tip"])),
                                                int(np.sum(asm["c_y"])),
                                                int(np.sum(asm["c_z"]))))
    print("  Σh_d              = %.12e W/K" % sum_hd)
    print("  Σh_d / h          = %.12e m^2   (串联热阻使其比 A_total 略小)"
          % (sum_hd / asm["h"]))
    print("  Σh_d/h vs A_total = %+.6e  (相对, 约 -h*d/(2k) 量级)"
          % ((sum_hd / asm["h"] - a_tot) / a_tot))
    print("  反推面积 ΣA_d      = %.12e m^2" % a_re)
    print("  A_total           = %.12e m^2" % a_tot)
    print("  相对差            = %+.6e" % rel)
    return check("R-2(a) 面积自洽 <1e-12", rel < 1e-12, "(%.3e)" % rel)


def r2b():
    hr("R-2 (b) 常温度场代数恒等式: T ≡ Tb, 逐行残差 vs 预期面通量")
    asm = F.assemble(BASE, 80, 12, 32)
    A, b = asm["A"], asm["b"]
    Tb, Tinf = asm["Tb"], asm["Tinf"]
    r = A @ np.full(asm["n"], Tb) - b
    e = ((asm["Hx"] * asm["c_tip"] + asm["Hy"] * asm["c_y"]
          + asm["Hz"] * asm["c_z"]) * (Tb - Tinf))
    d = r - e
    interior = ((asm["c_root"] == 0) & (asm["c_tip"] == 0)
                & (asm["c_y"] == 0) & (asm["c_z"] == 0))
    scale = float(np.max(np.abs(b)))
    rel = float(np.max(np.abs(d))) / scale
    print("  T≡Tb 时 a_P*Tb - Σa_nb*Tb - b 的预期值 = Σh_d*(Tb-Tinf) (根部项为 0)")
    print("  内部行数 %d / 总行数 %d" % (int(interior.sum()), asm["n"]))
    print("  内部行 max|r|        = %.6e   (应为 0: a_P 必须等于 Σa_nb)"
          % float(np.max(np.abs(r[interior]))))
    print("  全行 max|r - e|      = %.6e" % float(np.max(np.abs(d))))
    print("  max|b|               = %.6e" % scale)
    print("  相对残差 max|r-e|/max|b| = %+.6e" % rel)
    return check("R-2(b) 常温度场恒等式 <1e-12", rel < 1e-12, "(%.3e)" % rel)


def r2c():
    hr("R-2 (c) 有限 k 物理极限: k=1e5, 网格 80x12x32")
    p = dict(BASE, k=1.0e5)
    r = F.solve(p, 80, 12, 32, solver="direct")
    a_tot = 2 * 0.05 * 0.02 + 2 * 0.05 * 0.002 + 0.002 * 0.02
    q_exp = p["h"] * a_tot * (p["Tb"] - p["Tinf"])
    m = np.sqrt(2.0 * p["h"] / (p["k"] * p["t"]))
    pred = -(m * p["L"]) ** 2 / 3.0
    rel = r["Q"] / q_exp - 1.0
    print("  Qexp = h*A_total*dT = %.6f W" % q_exp)
    print("  Q                   = %.6f W" % r["Q"])
    print("  相对偏差            = %+.6e" % rel)
    print("  预期物理残差 -mL^2/3 = %+.6e   (mL = %.6f)" % (pred, m * p["L"]))
    print("  能量: |Qin-Qout| = %.6e W   |dQ|/Q = %.6e"
          % (r["abs_diff"], abs(r["imbalance"])))
    ok = check("R-2(c) k=1e5 物理极限 <1e-3", abs(rel) < 1e-3,
               "(%+.6e, 预期 %+.2e)" % (rel, pred))
    ok2 = a1("A-1 能量(k=1e5)", r)
    return ok, ok2


def r3():
    hr("R-3 网格体检: G1 / G2 (未知量、nnz、CSR、迭代、墙钟)")
    print("%-14s %10s %10s %9s %-10s %6s %9s %14s %12s"
          % ("grid", "N", "nnz", "CSR/MB", "solver", "iters", "wall/s",
             "Q/W (W/m)", "resid"))
    rows = []
    for ns in ((80, 12, 32), (160, 24, 64)):
        r = F.solve(BASE, *ns, solver="direct")
        g, s = r["grid"], r["solve"]
        rows.append(r)
        print("%-14s %10d %10d %9.2f %-10s %6d %9.2f %14.6f %12.2e"
              % ("%dx%dx%d" % (ns[0], ns[1], ns[2]), g["N"], g["nnz"],
                 g["csr_mb"], s["method"], s["iters"], s["wall_s"],
                 r["Q_per_span"], s["residual"]))
    n_ratio = rows[1]["grid"]["N"] / rows[0]["grid"]["N"]
    t_ratio = rows[1]["solve"]["wall_s"] / max(rows[0]["solve"]["wall_s"], 1e-9)
    dq = abs(rows[1]["Q_per_span"] - rows[0]["Q_per_span"]) / rows[1]["Q_per_span"]
    print("\n  N 比 G2/G1        = %.4f   (期望 ~8, 按体积加密)" % n_ratio)
    print("  墙钟比 G2/G1      = %.1f     (G1 太短时可忽略)" % t_ratio)
    print("  |Q2-Q1|/Q2        = %.3e" % dq)
    ok = check("R-3 未知量比 ~8", abs(n_ratio - 8.0) < 0.05, "(%.4f)" % n_ratio)
    for r in rows:
        a1("A-1 能量(%dx%dx%d)" % (r["grid"]["nx"], r["grid"]["ny"],
                                   r["grid"]["nz"]), r)
    return ok


# ---------------------------------------------------------------- E-2
def e2():
    hr("E-2  r=2 三向同步加密 (W=20mm): 七项体检 + 观察阶 p + GCI_fine")
    grids = [(80, 12, 32), (160, 24, 64), (320, 48, 128)]
    plans = [("direct", None), ("direct", None), ("cg", CG_RTOL)]
    print("  CG 相对容差写死 rtol = %.1e ; CG 后加 2 步残差回代(不放松 rtol)" % CG_RTOL)
    print("  L=%.4g m  t=%.4g m  W=%.4g m   (三向往同一 r=2 减半)"
          % (BASE["L"], BASE["t"], BASE["W"]))
    print("\n  %-3s %-12s %13s %13s %13s %8s %8s"
          % ("档", "网格", "dx/m", "dy/m", "dz/m", "dx/dy", "dz/dy"))
    geo = []
    for (nx, ny, nz) in grids:
        g = dict(dx=BASE["L"] / nx, dy=BASE["t"] / ny, dz=BASE["W"] / nz)
        geo.append(g)
        print("  L%-2d %-12s %13.6e %13.6e %13.6e %8.3f %8.3f"
              % (len(geo), "%dx%dx%d" % (nx, ny, nz), g["dx"], g["dy"], g["dz"],
                 g["dx"] / g["dy"], g["dz"] / g["dy"]))
    okar = check("E-2a Δx/Δy <= 4 (每档)",
                 all(g["dx"] / g["dy"] <= 4.0 for g in geo),
                 "(%.3f)" % (geo[0]["dx"] / geo[0]["dy"]))

    print("\n  %-12s %10s %10s %9s %-12s %6s %9s %14s %10s"
          % ("grid", "N", "nnz", "CSR/MB", "solver", "iters", "wall/s",
             "Q/W (W/m)", "resid"))
    qs = []
    for (nx, ny, nz), (md, rt) in zip(grids, plans):
        kw = dict(solver=md)
        if rt is not None:
            kw["rtol"] = rt
        r = F.solve(BASE, nx, ny, nz, **kw)
        g, s = r["grid"], r["solve"]
        qs.append(r["Q_per_span"])
        print("%-12s %10d %10d %9.2f %-12s %6d %9.2f %14.6f %10.2e"
              % ("%dx%dx%d" % (nx, ny, nz), g["N"], g["nnz"], g["csr_mb"],
                 s["method"], s["iters"], s["wall_s"], r["Q_per_span"],
                 s["residual"]))
        if s["method"] != "spsolve":
            qp = r["Q_pre"] if r["Q_pre"] is not None else r["Q"]
            print("             └ CG rtol=%.1e  resid_pre=%.2e  %d 步回代后 resid=%.2e"
                  "  迭代自检 |Q(cg)-Q(回代)|/Q=%.2e"
                  % (s["rtol"], s["resid_pre"], s["refine"], s["residual"],
                     abs(r["Q"] - qp) / abs(r["Q"])))
        a1("A-1 能量(%dx%dx%d)" % (nx, ny, nz), r)

    q1, q2, q3 = qs
    p_obs = np.log(abs((q2 - q1) / (q3 - q2))) / np.log(2.0)
    gci = 1.25 * abs(q3 - q2) / abs(q3) / (2.0 ** p_obs - 1.0)
    print("\n  Q1=%.6f  Q2=%.6f  Q3=%.6f  (W/m)" % (q1, q2, q3))
    print("  Q2-Q1=%.6e   Q3-Q2=%.6e" % (q2 - q1, q3 - q2))
    print("  观察阶 p = %.6f   (判据 [1.90, 2.10])" % p_obs)
    print("  GCI_fine = %.6f %%   (判据 < 0.5%%)" % (100.0 * gci))
    okp = check("E-2b p in [1.90,2.10]", 1.90 <= p_obs <= 2.10, "(p=%.4f)" % p_obs)
    okg = check("E-2c GCI_fine < 0.5%", gci < 5.0e-3, "(%.4f%%)" % (100 * gci))
    return okar and okp and okg


# ---------------------------------------------------------------- E-3
def e3():
    hr("E-3  1/W 外推 (基网格 80x12, Δz=0.625mm 固定, W=25t/50t/100t)")
    dz_fix = 6.25e-4
    print("  Δz = %.6e m 固定;  nx=80, ny=12" % dz_fix)
    print("\n  %10s %6s %10s %10s %12s %16s %14s %6s %9s %10s %10s"
          % ("W/m", "nz", "N", "dz/dy", "1/W (1/m)", "Q/W (W/m)",
             "Q(W)", "iters", "wall/s", "resid", "A-1 (W)"))
    xs, ys = [], []
    for ratio in (25.0, 50.0, 100.0):
        w = ratio * BASE["t"]
        nz = int(round(w / dz_fix))
        r = F.solve(dict(BASE, W=w), 80, 12, nz, solver="cg", rtol=CG_RTOL)
        xs.append(1.0 / w)
        ys.append(r["Q_per_span"])
        print("%10.6f %6d %10d %10.3f %12.4f %16.6f %14.6f %6d %9.2f %10.2e %10.2e"
              % (w, nz, r["grid"]["N"], (w / nz) / (BASE["t"] / 12), 1.0 / w,
                 r["Q_per_span"], r["Q"], r["solve"]["iters"],
                 r["solve"]["wall_s"], r["solve"]["residual"], r["abs_diff"]))
        a1("A-1 能量(W=%.0fmm)" % (1000 * w), r)

    xs, ys = np.array(xs), np.array(ys)
    amp = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(amp, ys, rcond=None)
    inter, slope = float(coef[0]), float(coef[1])
    dev = (inter - Q2D_REF) / Q2D_REF
    print("\n  1/W 线性外推:  a = %.6f W/m  (截距, 即 W->inf 的 Q/W)" % inter)
    print("                 b = %.6f W  (斜率, = 侧面对流附加总热量)" % slope)
    print("  2D 参考 Q2D_REF = %.6f W/m" % Q2D_REF)
    print("  相对偏差        = %+.6f %%" % (100 * dev))
    return check("E-3 外推截距 vs 210.888 <0.5%", abs(dev) < 5e-3,
                 "(%.4f%%)" % (100 * dev))


# ---------------------------------------------------------------- 标定 (S-1~S-4)
def cal():
    """本轮只跑 G2 一档, 报四个时间 + 迭代数 + rtol 实际值."""
    hr("标定 S: 只跑 G2 160x24x64, CG + ILU, 报装配/预条件/CG/总耗时")
    print("  spilu: drop_tol=%.1e  fill_factor=%.1f  permc_spec=MMD_AT_PLUS_A"
          % (F.DEFAULTS["spilu_drop"], F.DEFAULTS["spilu_fill"]))
    print("  CG 每 20 次迭代打印进度; 反代 2 步 (不放松 rtol)")
    print("  几何: L=%.4g t=%.4g W=%.4g  网格 160x24x64  (dx=%.6e dy=%.6e dz=%.6e)"
          % (BASE["L"], BASE["t"], BASE["W"],
             BASE["L"] / 160, BASE["t"] / 24, BASE["W"] / 64))

    rows = []
    for rt in (1.0e-10, 1.0e-12):
        hr("G2 一次求解  rtol = %.1e" % rt)
        r = F.solve(BASE, 160, 24, 64, solver="cg", rtol=rt, every=20)
        g, s = r["grid"], r["solve"]
        rows.append((rt, r, s))
        print("\n  网格 %dx%dx%d   N=%d   nnz=%d   CSR=%.2f MB"
              % (160, 24, 64, g["N"], g["nnz"], g["csr_mb"]))
        print("  求解器 %-14s  预条件 %-14s  CG 迭代 = %d  (info=%d, converged=%s)"
              % (s["method"], s["precond"], s["iters"], s["info"], s["converged"]))
        print("  rtol 实际值 = %.3e   resid_pre = %.3e   回代 %d 步后 resid = %.3e"
              % (s["rtol"], s["resid_pre"], s["refine"], s["residual"]))
        print("  ---- 四个计时点 (s, perf_counter 差值, 3 位小数) ----")
        print("  ① 装配  assembly      t_asm   = %8.3f" % s["t_asm"])
        print("  ② 预条件 precondition  t_prec  = %8.3f   (nnz(L+U)=%d)"
              % (s["t_prec"], s["fnz"]))
        print("  ③ 求解  solve         t_cg    = %8.3f   (含 %d 步回代 %.3f s)"
              % (s["t_cg"], s["refine"], s["t_ref"]))
        print("  ④ 后处理 postprocess   t_post  = %8.3f" % s["t_post"])
        print("  合计 t_total = %8.3f" % s["t_total"])
        print("  Q/W = %.6f W/m   Q = %.6f W   Q_pre = %s"
              % (r["Q_per_span"], r["Q"],
                 ("%.6f" % r["Q_pre"]) if r["Q_pre"] is not None else "n/a"))
        a1("A-1 能量(G2 rtol=%.0e)" % rt, r)

    hr("G2 标定汇总")
    print("  %-10s %8s %10s %10s %10s %10s %14s %12s"
          % ("rtol", "iters", "t_asm/s", "t_prec/s", "t_cg/s", "t_post/s",
             "Q/W (W/m)", "resid"))
    for rt, r, s in rows:
        print("  %-10.1e %8d %10.3f %10.3f %10.3f %10.3f %14.6f %12.2e"
              % (rt, s["iters"], s["t_asm"], s["t_prec"], s["t_cg"],
                 s["t_post"], r["Q_per_span"], s["residual"]))
    dq = abs(rows[0][1]["Q_per_span"] - rows[1][1]["Q_per_span"])
    print("  |Q(1e-10) - Q(1e-12)| / Q = %.3e   (判阶时 rtol=1e-10 是否够用的实测依据)"
          % (dq / abs(rows[1][1]["Q_per_span"])))
    print("  参考: 上一轮 spsolve 同网格 Q/W = 228.170561, wall = 348.91 s")
    return not FAILS


# ---------------------------------------------------------------- 小网格全链路 (D-2)
def smoke():
    """40x6x16 = 3840 未知量: 验证 CG+ILU 全链路 + 四个计时点 + 与 spsolve 互校."""
    hr("D-2 小网格全链路: 40x6x16 = 3840 未知量, CG + ILU")
    nx, ny, nz = 40, 6, 16
    print("  几何 L=%.4g t=%.4g W=%.4g  →  dx=%.6e dy=%.6e dz=%.6e"
          % (BASE["L"], BASE["t"], BASE["W"],
             BASE["L"] / nx, BASE["t"] / ny, BASE["W"] / nz))
    print("  spilu: drop_tol=%.1e  fill_factor=%.1f  permc_spec=%s"
          % (F.DEFAULTS["spilu_drop"], F.DEFAULTS["spilu_fill"],
             F.DEFAULTS["spilu_perm"]))

    hr("40x6x16  CG + ILU  (rtol=1e-12)")
    r = F.solve(BASE, nx, ny, nz, solver="cg", rtol=1.0e-12, every=20)
    g, s = r["grid"], r["solve"]
    print("\n  ① 装配能跑通  N=%d  nnz=%d  CSR=%.2f MB" % (g["N"], g["nnz"], g["csr_mb"]))
    print("  ② spilu 能跑通  预条件 = %-16s  nnz(L+U) = %d" % (s["precond"], s["fnz"]))
    print("  ③ CG 收敛     iters = %d  info = %d  converged = %s  resid = %.3e"
          % (s["iters"], s["info"], s["converged"], s["residual"]))
    print("  ④ 四个计时点:")
    print("     ① 装配  assembly      t_asm  = %8.3f s" % s["t_asm"])
    print("     ② 预条件 precondition  t_prec = %8.3f s" % s["t_prec"])
    print("     ③ 求解  solve         t_cg   = %8.3f s" % s["t_cg"])
    print("     ④ 后处理 postprocess   t_post = %8.3f s" % s["t_post"])
    print("     合计                    t_total= %8.3f s" % s["t_total"])
    print("  rtol 实际值 = %.3e   resid_pre = %.3e   回代 %d 步"
          % (s["rtol"], s["resid_pre"], s["refine"]))
    print("  Q/W = %.6f W/m   Q = %.6f W" % (r["Q_per_span"], r["Q"]))

    hr("40x6x16  同网格 spsolve 互校")
    rd = F.solve(BASE, nx, ny, nz, solver="direct")
    sd = rd["solve"]
    dq = abs(r["Q_per_span"] - rd["Q_per_span"])
    rel = dq / abs(rd["Q_per_span"])
    print("  spsolve Q/W = %.6f   iters=%d   t_total=%.3f s"
          % (rd["Q_per_span"], sd["iters"], sd["t_total"]))
    print("  |Q(cg) - Q(spsolve)| / Q = %.3e" % rel)
    ok = check("D-2 小网格 CG+ILU 全链路",
               s["converged"] and s["iters"] > 0 and s["fnz"] > 0 and rel < 1e-8,
               "(iters=%d, |dQ|/Q=%.2e)" % (s["iters"], rel))
    a1("A-1 能量(40x6x16 cg)", r)
    a1("A-1 能量(40x6x16 spsolve)", rd)
    return ok


# ---------------------------------------------------------------- R-2 迭代解互校
def pair():
    """R-2: 40x6x16 与 80x12x32 两档, GMRES+ILU(0) 与 CG+Jacobi 各跑一次,
    与同网格 spsolve 比, 要求 |Q_iter - Q_spsolve|/Q_spsolve < 1e-10."""
    hr("R-2 迭代解与直接解互校 (两档 x 两种配对)")
    print("  R-1 配对: CG→Jacobi   GMRES→ILU(0)   (混合则 solve() 直接报错)")
    print("  rtol = 1e-12 (与 spsolve 比对的判据为 |dQ|/Q < 1e-10)")
    print("  spilu: drop_tol=%.1f  fill_factor=%.1f  permc_spec=%s"
          % (F.DEFAULTS["spilu_drop"], F.DEFAULTS["spilu_fill"],
             F.DEFAULTS["spilu_perm"]))
    tol = 1.0e-10
    allok = True

    for (nx, ny, nz) in ((40, 6, 16), (80, 12, 32)):
        tag = "%dx%dx%d" % (nx, ny, nz)
        hr("%s  spsolve 基准" % tag)
        rd = F.solve(BASE, nx, ny, nz, solver="direct")
        gd, sd = rd["grid"], rd["solve"]
        print("  N=%d  nnz=%d  CSR=%.2f MB" % (gd["N"], gd["nnz"], gd["csr_mb"]))
        print("  Q/W = %.12f W/m   Q = %.12f W" % (rd["Q_per_span"], rd["Q"]))
        print("  四个计时点: t_asm=%.3f  t_prec=%.3f  t_solve=%.3f  t_post=%.3f  total=%.3f s"
              % (sd["t_asm"], sd["t_prec"], sd["t_cg"], sd["t_post"], sd["t_total"]))
        a1("A-1 能量(%s spsolve)" % tag, rd)

        for mode, pk, lab in (("gmres", "ilu", "GMRES+ILU(0)"),
                              ("cg", "jacobi", "CG+Jacobi")):
            hr("%s  %s" % (tag, lab))
            try:
                r = F.solve(BASE, nx, ny, nz, solver=mode, precond=pk,
                            rtol=1e-12, every=20)
            except Exception as exc:
                print("  [ERROR] %s: %s" % (type(exc).__name__, exc))
                print("[FAIL]  R-2 %s %s" % (tag, lab))
                FAILS.append("R-2 %s %s" % (tag, lab))
                allok = False
                continue
            s = r["solve"]
            rel = abs(r["Q_per_span"] - rd["Q_per_span"]) / abs(rd["Q_per_span"])
            print("  求解器 %-16s 预条件 %-10s 迭代 = %d  converged = %s"
                  % (s["method"], s["precond"], s["iters"], s["converged"]))
            print("  rtol 实际值 = %.1e   resid_pre = %.3e   回代 %d 步后 resid = %.3e"
                  % (s["rtol"], s["resid_pre"], s["refine"], s["residual"]))
            print("  四个计时点: t_asm=%.3f  t_prec=%.3f  t_solve=%.3f  t_post=%.3f  total=%.3f s"
                  % (s["t_asm"], s["t_prec"], s["t_cg"], s["t_post"], s["t_total"]))
            print("  Q/W = %.12f W/m   (spsolve %.12f)   |dQ|/Q = %.3e"
                  % (r["Q_per_span"], rd["Q_per_span"], rel))
            a1("A-1 能量(%s %s)" % (tag, lab), r)
            ok = check("R-2 %s %s  vs spsolve <1e-10" % (tag, lab), rel < tol,
                       "(%.3e)" % rel)
            allok = allok and ok

    hr("R-2 汇总")
    print("  两档 x 两种配对 全部 |dQ|/Q < 1e-10 才能上 G3")
    return allok


# ---------------------------------------------------------------- R-6-2 LU 填充标定
def lu_fill():
    """R-6-2: G1/G2 各一次 splu(完整 LU), 报 nnz(L+U)/nnz(A) 与内存,
    再用两档实测指数外推 G3(320x48x128, N=1,966,080) 的内存."""
    hr("R-6-2  LU 填充标定: G1 / G2 各一次 splu(direct), 报填充比与内存")
    print("  splu 参数: permc_spec=COLAMD, diag_pivot_thresh=1.0, 其余默认")
    print("  内存口径: 12 B/nnz (8 B double 数值 + 4 B int32 行号) + 4 B*(N+1) 列指针")
    print("            另加输入矩阵本身 CSR 占用, 一并列出")
    print("\n  %-14s %10s %11s %13s %8s %10s %10s %9s"
          % ("grid", "N", "nnz(A)", "nnz(L+U)", "fill", "t_splu/s",
             "LU/MB", "CSR/MB"))
    rows = []
    for ns in ((80, 12, 32), (160, 24, 64)):
        nx, ny, nz = ns
        tag = "%dx%dx%d" % ns
        asm = F.assemble(BASE, nx, ny, nz)
        A = asm["A"].tocsc()
        n = int(asm["n"])
        t0 = time.perf_counter()
        try:
            lu = spla.splu(A)
        except Exception as exc:
            print("%-14s %10d %11d  [SPLU FAILED] %s: %s"
                  % (tag, n, asm["nnz"], type(exc).__name__, exc))
            FAILS.append("R-6-2 splu %s" % tag)
            continue
        dt = time.perf_counter() - t0
        fnz = int(lu.L.nnz + lu.U.nnz)
        fill = fnz / float(asm["nnz"])
        mem_mb = fnz * 12.0 / 1048576.0 + (n + 1) * 4.0 / 1048576.0
        print("%-14s %10d %11d %13d %8.2f %10.3f %10.1f %9.1f"
              % (tag, n, asm["nnz"], fnz, fill, dt, mem_mb, asm["csr_mb"]))
        rows.append(dict(tag=tag, n=n, nnza=int(asm["nnz"]), fnz=fnz,
                         fill=fill, t=dt, mem_mb=mem_mb, csr=asm["csr_mb"]))
        del lu, A

    hr("R-6-2  G3 外推 (1,966,080 未知量)")
    if len(rows) < 2:
        print("  [STOP] 实测档位不足两个, 无法拟合填充指数; 不提交 G3, 内存待定")
        FAILS.append("R-6-2 外推数据不足")
        return False

    n1, n2 = rows[0]["n"], rows[1]["n"]
    f1, f2 = float(rows[0]["fnz"]), float(rows[1]["fnz"])
    pexp = float(np.log(f2 / f1) / np.log(float(n2) / float(n1)))
    n3 = 320 * 48 * 128
    f3 = f2 * (float(n3) / float(n2)) ** pexp
    mem3_mb = f3 * 12.0 / 1048576.0 + (n3 + 1) * 4.0 / 1048576.0
    mem3_csr = rows[1]["csr"] * (float(n3) / float(n2))
    tot_gb = (mem3_mb + mem3_csr) / 1024.0

    print("  G1 nnz(L+U) = %d (N=%d)" % (rows[0]["fnz"], n1))
    print("  G2 nnz(L+U) = %d (N=%d)" % (rows[1]["fnz"], n2))
    print("  实测填充指数 p = ln[nnz3/nnz2] / ln[N3/N2] 的两档拟合 = %.4f" % pexp)
    print("  (三维嵌套剖分理论 O(N^4/3) → p = 1.3333; 自然序 O(N^2) → p = 2.0)")
    print("  外推 G3 N = %d  nnz(L+U) = %.4e  LU 内存 = %.2f GB   CSR = %.2f GB"
          % (n3, f3, mem3_mb / 1024.0, mem3_csr / 1024.0))
    print("  合计峰值（粗估）: %.2f GB" % tot_gb)
    ok = tot_gb < 16.0
    check("R-7-2 G3 外推内存 < 16 GB", ok, "(%.2f GB, p=%.3f)" % (tot_gb, pexp))
    if not ok:
        print("  → 内存待定: 不提交 G3 spsolve")
    return ok


# ---------------------------------------------------------------- G3 双 rtol 自证
def g3():
    """G3 320x48x128 (N=1,966,080)  CG+Jacobi, 同一装配跑多个 rtol 自证收敛.
    判据: 相邻两档 |dQ|/Q < 1e-10."""
    hr("G3 320x48x128 (N=1,966,080)  CG+Jacobi  双 rtol 自证")
    nx, ny, nz = 320, 48, 128
    dx, dy, dz = BASE["L"] / nx, BASE["t"] / ny, BASE["W"] / nz
    print("  N = %d" % (nx * ny * nz))
    print("  dx=%.6e  dy=%.6e  dz=%.6e" % (dx, dy, dz))
    print("  Δx/Δy = %.4f   Δz/Δy = %.4f   (判据 <= 4)" % (dx / dy, dz / dy))
    print("  预条件写死 Jacobi (R-1); 回代 refine = %d 步; maxiter = %d"
          % (F.DEFAULTS["refine"], F.DEFAULTS["maxiter"]))
    check("G3 长宽比 Δx/Δy <= 4", (dx / dy) <= 4.0, "(%.4f)" % (dx / dy))

    rows = []
    ladders = (1.0e-12, 1.0e-13, 1.0e-14)
    passed = False
    for rt in ladders:
        hr("G3  rtol = %.0e" % rt)
        r = F.solve(BASE, nx, ny, nz, solver="cg", precond="jacobi",
                    rtol=rt, every=200)
        g, s = r["grid"], r["solve"]
        rows.append((rt, r, s))
        print("  网格 %dx%dx%d   N=%d   nnz=%d   CSR=%.1f MB"
              % (nx, ny, nz, g["N"], g["nnz"], g["csr_mb"]))
        print("  求解器 %-12s 预条件 %-8s 迭代 = %d  info = %d  converged = %s"
              % (s["method"], s["precond"], s["iters"], s["info"],
                 s["converged"]))
        print("  rtol 实际值 = %.1e   resid_pre = %.3e   回代 %d 步后 resid = %.3e"
              % (s["rtol"], s["resid_pre"], s["refine"], s["residual"]))
        print("  ---- 四个计时点 (s) ----")
        print("  ① 装配   t_asm   = %10.3f" % s["t_asm"])
        print("  ② 预条件 t_prec  = %10.3f" % s["t_prec"])
        print("  ③ 求解   t_solve = %10.3f   (含回代 %.3f)" % (s["t_cg"], s["t_ref"]))
        print("  ④ 后处理 t_post  = %10.3f   total = %.3f" % (s["t_post"], s["t_total"]))
        print("  Q/W = %.6f W/m    Q = %.6f W" % (r["Q_per_span"], r["Q"]))
        a1("A-1 能量(G3 rtol=%.0e)" % rt, r)

        if len(rows) >= 2:
            q1 = rows[-2][1]["Q_per_span"]
            q2 = rows[-1][1]["Q_per_span"]
            rel = abs(q2 - q1) / abs(q2)
            ok = rel < 1e-10
            passed = passed or ok
            print("  |dQ|/Q  = |%.6f - %.6f| / %.6f = %.3e" % (q2, q1, q2, rel))
            check("G3 数值收敛 |dQ|/Q < 1e-10  (%.1e vs %.1e)"
                  % (rows[-2][0], rows[-1][0]), ok, "(%.3e)" % rel)
            if ok:
                print("  → 已达到 1e-10, 不再降 rtol")
                break
            print("  → 未达 1e-10, 降一档 rtol 重跑")

    hr("G3 汇总")
    for rt, r, s in rows:
        print("  rtol=%.0e  iters=%6d  converged=%-5s  t_total=%8.3f s  Q=%.6f W"
              % (rt, s["iters"], s["converged"], s["t_total"], r["Q"]))
    print("  数值收敛判定: %s" % ("PASS" if passed else "FAIL (全部档位均未达 1e-10)"))
    return passed


# ---------------------------------------------------------------- E-3 精简版
def e3b():
    """E-3 精简版: W=20mm 沿用 G3 结果, 补 W=50mm / W=100mm 两档, 单 rtol=1e-12.
    Δz 固定 1.5625e-4, 即 W=50 → Nz=320, W=100 → Nz=640.
    拟合 Q/W = a*(1/W) + b, 判 |b - 210.888|/210.888 < 0.5%."""
    hr("E-3 精简版: 补 W=50mm / W=100mm, Δz = 1.5625e-4 固定, 单 rtol=1e-12")
    print("  W=20mm 档沿用 G3 结果, 不重跑: Q/W = 228.171306 W/m (仅 6 位小数可用)")
    print("  预条件: Jacobi (R-1); refine = %d; maxiter = %d"
          % (F.DEFAULTS["refine"], F.DEFAULTS["maxiter"]))

    xs = [1.0 / 0.020]
    ys = [228.171306]
    tags = ["W=20mm"]

    for W, nz in ((0.050, 320), (0.100, 640)):
        nx, ny = 320, 48
        tag = "W=%dmm" % int(round(W * 1000.0))
        dz = W / nz
        hr("%s   Nx=%d Ny=%d Nz=%d" % (tag, nx, ny, nz))
        print("  dx=%.6e  dy=%.6e  dz=%.6e   Δx/Δy=%.4f  Δz/Δy=%.4f"
              % (BASE["L"] / nx, BASE["t"] / ny, dz,
                 (BASE["L"] / nx) / (BASE["t"] / ny), dz / (BASE["t"] / ny)))
        try:
            r = F.solve(dict(BASE, W=W), nx, ny, nz, solver="cg",
                        precond="jacobi", rtol=1e-12, every=500)
        except Exception as exc:
            print("  [ERROR] %s: %s" % (type(exc).__name__, exc))
            FAILS.append("E-3 %s" % tag)
            continue
        g, s = r["grid"], r["solve"]
        print("  N = %d   nnz = %d   CSR = %.2f MB"
              % (g["N"], g["nnz"], g["csr_mb"]))
        print("  迭代 = %d   info = %d   converged = %s"
              % (s["iters"], s["info"], s["converged"]))
        print("  t_asm = %.3f   t_prec = %.3f   t_solve = %.3f   t_post = %.3f"
              "   total = %.3f s"
              % (s["t_asm"], s["t_prec"], s["t_cg"], s["t_post"], s["t_total"]))
        print("  resid_pre = %.3e   回代 %d 步后 resid = %.3e"
              % (s["resid_pre"], s["refine"], s["residual"]))
        print("  Q/W = %.9f W/m    Q = %.9f W" % (r["Q_per_span"], r["Q"]))
        a1("A-1 能量(%s)" % tag, r)
        check("E-3 %s converged" % tag, s["converged"],
              "(iters=%d, resid=%.2e)" % (s["iters"], s["residual"]))
        xs.append(1.0 / W)
        ys.append(r["Q_per_span"])
        tags.append(tag)

    hr("E-3 线性拟合   Q/W = a*(1/W) + b")
    if len(xs) < 3:
        print("  [STOP] 可用点不足 3 个 (%d), 不做拟合" % len(xs))
        FAILS.append("E-3 拟合点数不足")
        return False

    x = np.array(xs, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    rel = abs(b - Q2D_REF) / Q2D_REF

    print("  %-9s %14s %16s %16s %14s"
          % ("档位", "1/W (1/m)", "Q/W 实测", "Q/W 拟合", "残差"))
    for tg, xi, yi, fi in zip(tags, x, y, yhat):
        print("  %-9s %14.9f %16.9f %16.9f %+14.3e"
              % (tg, xi, yi, fi, yi - fi))
    print("")
    print("  a = %.9f   (W/m per 1/m)" % a)
    print("  b = %.9f   W/m" % b)
    print("  R^2 = %.12f" % r2)
    print("  2D 参考 Q2D_REF = %.3f W/m" % Q2D_REF)
    print("  |b - Q2D_REF| / Q2D_REF = %.6e  (%.4f%%)"
          % (rel, 100.0 * rel))
    print("  备注: W=20mm 点只有 6 位小数 (228.171306), 拟合结果受此限制")
    return check("E-3 |b-210.888|/210.888 < 0.5%", rel < 5.0e-3,
                 "(b=%.6f, %.4f%%)" % (b, 100.0 * rel))


# ---------------------------------------------------------------- main
def main():
    stage = (sys.argv[1] if len(sys.argv) > 1 else "e1").lower()
    hr("fin3d verification (A-1 终版 / R-2 / R-3 / E-1,E-2,E-3)  numpy %s  scipy %s  stage=%s"
       % (np.__version__, __import__("scipy").__version__, stage))
    print("base: L=%.4g t=%.4g W=%.4g k=%.6g h=%.6g Tb=%.6g Tinf=%.6g"
          % (BASE["L"], BASE["t"], BASE["W"], BASE["k"], BASE["h"],
             BASE["Tb"], BASE["Tinf"]))
    print("判据: E-1(1)<0.1%  R-2(a)<1e-12  R-2(b)<1e-12  R-2(c)<1e-3  "
          "A-1 <= max(1e-9*Q, 1e-6 W)")
    print("      E-2  p in [1.90,2.10], GCI_fine<0.5%   E-3  |a-210.888|/210.888<0.5%")

    if stage in ("cal", "smoke", "pair", "lu", "g3", "e3b"):
        if stage == "smoke":
            smoke()
        elif stage == "pair":
            pair()
        elif stage == "lu":
            lu_fill()
        elif stage == "g3":
            g3()
        elif stage == "e3b":
            e3b()
        else:
            cal()
        print("\n  总墙钟 = %.1f s" % (time.perf_counter() - T0))
        print("FINAL: %s" % (("PASS(%s)" % stage) if not FAILS
                             else "FAIL (" + ", ".join(FAILS) + ")"))
        return 0 if not FAILS else 1

    ok1, oka = gate_e1()
    oka2 = r2a()
    okb = r2b()
    okc, okc2 = r2c()
    okr3 = r3()

    hr("E-1 阶段汇总")
    for tag, ok in (("E-1(1) 一维退化", ok1), ("A-1 能量(1D)", oka),
                    ("R-2(a) 面积自洽", oka2), ("R-2(b) 常温度场", okb),
                    ("R-2(c) k=1e5", okc), ("A-1 能量(k=1e5)", okc2),
                    ("R-3 网格体检", okr3)):
        print("  %-20s %s" % (tag, "PASS" if ok else "FAIL"))
    print("\n  累计墙钟 = %.1f s" % (time.perf_counter() - T0))
    if FAILS:
        print("E-1 GATE: FAIL  (%s)" % ", ".join(FAILS))
        print("FINAL: FAIL")
        return 1
    print("E-1 GATE: PASS")

    if stage == "e1":
        print("stage=e1: 停在 E-1")
        print("FINAL: PASS(e1)")
        return 0

    e1_ok = True
    if stage in ("all", "e2"):
        if not e2():
            e1_ok = False
    if stage in ("all", "e3"):
        if not e3():
            e1_ok = False
    print("\n  总墙钟 = %.1f s" % (time.perf_counter() - T0))
    print("FINAL: %s" % ("PASS" if (e1_ok and not FAILS)
                         else "FAIL (" + ", ".join(FAILS) + ")"))
    return 0 if (e1_ok and not FAILS) else 1


if __name__ == "__main__":
    sys.exit(main())
