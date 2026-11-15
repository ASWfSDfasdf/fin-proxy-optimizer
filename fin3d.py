#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fin3d.py —— 三维矩形肋片稳态导热的格心有限体积求解器.

几何 : L (x, 流向) x t (y, 厚度) x W (z, 展向)
边界 : x=0 根部定温 Tb; x=L 端面, y=0/t 上下表面, z=0/W 两侧面 对流 h
拓扑 : 格心型 —— 未知量在控制体中心, 域边界与格子面重合, 无 1/4 (1/8) 格
离散 : 7 点格式, 对流面用串联热阻 A/(1/h + d/(2k)), 二阶
求解 : scipy.sparse 稀疏直接解 (小网格) 或 CG + ILU(0) (大网格)

接口
  assemble(params, nx, ny, nz) -> dict(A, b, diag, offsum, Hx..., 供验收脚本核对)
  solve(params, nx, ny, nz)    -> dict(Q, Q_per_span, Q_out, fluxes, ...)

系数约定 (a_P*T_P = sum(a_nb*T_nb) + b):
  Gx = k*dy*dz/dx, Gy = k*dx*dz/dy, Gz = k*dx*dy/dz            内部面
  Hx = Ax/(1/h + dx/(2k)), Hy, Hz                              对流面(串联热阻)
  Rx = 2*k*Ax/dx = k*Ax/(dx/2)                                 根部定温(半格导热)
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

DEFAULTS = dict(L=0.05, t=0.002, W=0.02, k=200.0, h=50.0,
                Tb=350.0, Tinf=300.0, z_bc="h", solver="auto",
                rtol=1e-11, maxiter=20000, direct_limit=400000, refine=2,
                spilu_drop=0.0, spilu_fill=1.0, spilu_perm="COLAMD",
                restart=200, precond="auto")

ADIABATIC = ("adiabatic", "adiab", "sym", "insulated", "none")


def _masks(nx, ny, nz):
    """线性索引 p = (k*ny + j)*nx + i 下的三维下标."""
    n = nx * ny * nz
    p = np.arange(n, dtype=np.int64)
    return p, p % nx, (p // nx) % ny, p // (nx * ny)


def assemble(params, nx, ny, nz):
    """装配 7 点矩阵与右端, 返回 dict (含逐面传导, 供几何核对)."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in dict(params).items() if v is not None})
    nx, ny, nz = int(nx), int(ny), int(nz)
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx/ny/nz 必须 >= 1")

    L, t, W = float(p["L"]), float(p["t"]), float(p["W"])
    kc, h = float(p["k"]), float(p["h"])
    Tb, Tinf = float(p["Tb"]), float(p["Tinf"])
    z_conv = str(p["z_bc"]).lower() not in ADIABATIC

    dx, dy, dz = L / nx, t / ny, W / nz
    Ax, Ay, Az = dy * dz, dx * dz, dx * dy          # x/y/z 法向面面积
    Gx, Gy, Gz = kc * Ax / dx, kc * Ay / dy, kc * Az / dz
    Hx = Ax / (1.0 / h + dx / (2.0 * kc))
    Hy = Ay / (1.0 / h + dy / (2.0 * kc))
    Hz = Az / (1.0 / h + dz / (2.0 * kc))
    Rx = 2.0 * kc * Ax / dx

    I, i_, j_, k_ = _masks(nx, ny, nz)
    n = I.size
    diag = np.zeros(n)
    rhs = np.zeros(n)
    offsum = np.zeros(n)
    rows, cols, vals = [], [], []

    for off, g, has in ((1, Gx, i_ < nx - 1), (nx, Gy, j_ < ny - 1),
                        (nx * ny, Gz, k_ < nz - 1)):
        back = np.zeros_like(has)
        back[off:] = has[:-off]
        if not (has.any() or back.any()):
            continue
        offsum += g * (has.astype(np.float64) + back.astype(np.float64))
        if has.any():
            rows.append(I[has]); cols.append(I[has] + off)
            vals.append(np.full(int(has.sum()), -g))
        if back.any():
            rows.append(I[back]); cols.append(I[back] - off)
            vals.append(np.full(int(back.sum()), -g))
    diag += offsum

    c_root = (i_ == 0).astype(np.float64)
    c_tip = (i_ == nx - 1).astype(np.float64)
    c_y = (j_ == 0).astype(np.float64) + (j_ == ny - 1).astype(np.float64)
    c_z = np.zeros(n)
    if z_conv:
        c_z = (k_ == 0).astype(np.float64) + (k_ == nz - 1).astype(np.float64)

    diag += Rx * c_root + Hx * c_tip + Hy * c_y + Hz * c_z
    rhs += (Rx * Tb * c_root + Hx * Tinf * c_tip
            + Hy * Tinf * c_y + Hz * Tinf * c_z)

    rows.append(I); cols.append(I); vals.append(diag)
    A = sp.coo_matrix((np.concatenate(vals),
                       (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n)).tocsr()
    return dict(A=A, b=rhs, diag=diag, offsum=offsum, params=p,
                dx=dx, dy=dy, dz=dz, Ax=Ax, Ay=Ay, Az=Az,
                Gx=Gx, Gy=Gy, Gz=Gz, Hx=Hx, Hy=Hy, Hz=Hz, Rx=Rx,
                c_root=c_root, c_tip=c_tip, c_y=c_y, c_z=c_z,
                z_conv=z_conv, n=n, nnz=int(A.nnz), nx=nx, ny=ny, nz=nz,
                h=h, kc=kc, Tb=Tb, Tinf=Tinf, L=L, t=t, W=W,
                csr_mb=float((A.data.nbytes + A.indices.nbytes +
                              A.indptr.nbytes) / 1048576.0))


def _bicgstab(a, b, rtol, maxiter, m, every=0, t0=None):
    """BiCGSTAB (可与 ILU(0) 配对), 返回 (x, info, 实际迭代数)."""
    n_iter = [0]
    t0 = time.perf_counter() if t0 is None else t0
    bn = float(np.linalg.norm(b))

    def _cb(_xk):
        n_iter[0] += 1
        if every and n_iter[0] % int(every) == 0:
            rr = float(np.linalg.norm(b - a @ _xk)) / max(bn, 1e-300)
            print("    [bicg] iter=%6d  |r|/|b| = %.3e  t = %8.2f s"
                  % (n_iter[0], rr, time.perf_counter() - t0), flush=True)

    try:
        x, info = spla.bicgstab(a, b, rtol=rtol, maxiter=maxiter, M=m, callback=_cb)
    except TypeError:
        x, info = spla.bicgstab(a, b, tol=rtol, maxiter=maxiter, M=m, callback=_cb)
    return x, info, n_iter[0]


def _gmres(a, b, rtol, maxiter, m, restart=200, every=0, t0=None):
    """GMRES(restart) (可与 ILU(0)/ILU(k) 配对), 返回 (x, info, 实际迭代数).

    scipy 的 maxiter 计作内迭代总数(retcode 与重启次数无关).
    callback_type="x" 时可算真残差 |b-Ax|; 旧版 scipy 退回 pr_norm 计数.
    """
    n_iter = [0]
    t0 = time.perf_counter() if t0 is None else t0
    bn = float(np.linalg.norm(b))

    def _cb_x(_xk):
        n_iter[0] += 1
        if every and n_iter[0] % int(every) == 0:
            rr = float(np.linalg.norm(b - a @ _xk)) / max(bn, 1e-300)
            print("    [gmres] iter=%6d  |r|/|b| = %.3e  t = %8.2f s"
                  % (n_iter[0], rr, time.perf_counter() - t0), flush=True)

    def _cb_pr(_pr):
        n_iter[0] += 1
        if every and n_iter[0] % int(every) == 0:
            print("    [gmres] iter=%6d  pr_norm   = %.3e  t = %8.2f s"
                  % (n_iter[0], float(_pr), time.perf_counter() - t0), flush=True)

    try:
        x, info = spla.gmres(a, b, rtol=rtol, maxiter=maxiter, M=m,
                             restart=restart, callback=_cb_x,
                             callback_type="x")
    except TypeError:
        x, info = spla.gmres(a, b, tol=rtol, maxiter=maxiter, M=m,
                             restart=restart, callback=_cb_pr)
    return x, info, n_iter[0]


def _cg(a, b, rtol, maxiter, m, every=0, t0=None):
    """兼容 scipy 版本差异的 CG, 返回 (x, info, 实际迭代数).

    every>0 时每 every 次迭代打印 iter / |r|/|b| / 已用秒数 (S-4 进度可观测).
    """
    n_iter = [0]
    t0 = time.perf_counter() if t0 is None else t0
    bn = float(np.linalg.norm(b))

    def _cb(_xk):
        n_iter[0] += 1
        if every and n_iter[0] % int(every) == 0:
            rr = float(np.linalg.norm(b - a @ _xk)) / max(bn, 1e-300)
            print("    [cg] iter=%6d  |r|/|b| = %.3e  t = %8.2f s"
                  % (n_iter[0], rr, time.perf_counter() - t0), flush=True)

    try:
        x, info = spla.cg(a, b, rtol=rtol, maxiter=maxiter, M=m, callback=_cb)
    except TypeError:                                    # scipy < 1.12
        x, info = spla.cg(a, b, tol=rtol, maxiter=maxiter, M=m, callback=_cb)
    return x, info, n_iter[0]


def solve(params, nx, ny, nz, solver=None, rtol=None, refine=None,
          verbose=False, every=0, precond=None):
    """求解一个算例, 返回 dict(Q, Q_per_span, T, fluxes, grid, solve).

    refine>0: CG 之后再做 refine 步残差回代 x += M^-1 (b - A x).
    用途 —— 残差 1e-12 不代表解误差 1e-12 (后者可被条件数放大), 而 E-2 的
    Q3-Q2 只有 1e-6 相对量级, 必须把迭代误差压到差分噪声以下.
    只做回代, 不放松 rtol.
    """
    t_a0 = time.perf_counter()
    asm = assemble(params, nx, ny, nz)
    t_asm = time.perf_counter() - t_a0
    A, rhs, n = asm["A"], asm["b"], asm["n"]
    p = asm["params"]
    mode = str(solver or p["solver"]).lower()
    if precond is not None:
        p["precond"] = precond
    rtol = float(p["rtol"] if rtol is None else rtol)

    t0 = time.perf_counter()
    t_prec = t_cg = 0.0
    fnz = 0
    if mode in ("direct", "spsolve") or (mode == "auto" and n <= int(p["direct_limit"])):
        used, precond, M = "spsolve", "none", None
        x = spla.spsolve(A.tocsc(), rhs)
        info, n_iter = 0, 0
    else:
        if mode == "auto":
            mode = "cg"
        pk = str(p["precond"]).lower()
        if pk in ("auto", ""):
            pk = "ilu" if mode in ("gmres", "bicgstab") else "jacobi"
        # R-1 配对规则: 混搭直接报错, 不在运行时默默换掉
        if mode == "cg" and pk not in ("jacobi",):
            raise ValueError("R-1: CG 只能配 Jacobi/SSOR/IC(0); 实测 ILU 非对称会使 CG 停滞. 当前 pk=%s" % pk)
        if mode in ("gmres", "bicgstab") and pk not in ("ilu",):
            raise ValueError("R-1: %s 只能配 ILU(0)/ILU(k); 当前 pk=%s" % (mode, pk))

        drop = float(p["spilu_drop"])
        fill = float(p["spilu_fill"])
        perm = str(p["spilu_perm"])
        t_p0 = time.perf_counter()
        if pk == "ilu":
            print("  [ilu] start %s  nnz(A)=%d  drop_tol=%g  fill_factor=%g"
                  "  permc_spec=%s" % (time.strftime("%H:%M:%S"), asm["nnz"],
                                      drop, fill, perm), flush=True)
            try:
                ilu = spla.spilu(A.tocsc(), drop_tol=drop, fill_factor=fill,
                                 permc_spec=perm)
                M = spla.LinearOperator((n, n), ilu.solve)
                precond = "ILU(%g/%g)" % (drop, fill)
                try:
                    fnz = int(ilu.L.nnz + ilu.U.nnz)
                except Exception:
                    fnz = -1
            except Exception as exc:
                raise RuntimeError("spilu 失败: %s" % exc)
            print("  [ilu] end   %s  elapsed = %.3f s  nnz(L+U)=%d"
                  % (time.strftime("%H:%M:%S"), time.perf_counter() - t_p0, fnz),
                  flush=True)
        else:                                            # jacobi
            dinv = 1.0 / A.diagonal()
            M = spla.LinearOperator((n, n), lambda v: v * dinv)
            precond = "Jacobi"
            fnz = n
        t_prec = time.perf_counter() - t_p0

        used = mode + "+" + precond
        t_c0 = time.perf_counter()
        if mode == "gmres":
            x, info, n_iter = _gmres(A, rhs, rtol, int(p["maxiter"]), M,
                                     restart=int(p["restart"]),
                                     every=every, t0=t_c0)
        elif mode == "bicgstab":
            x, info, n_iter = _bicgstab(A, rhs, rtol, int(p["maxiter"]), M,
                                        every=every, t0=t_c0)
        else:
            x, info, n_iter = _cg(A, rhs, rtol, int(p["maxiter"]), M,
                                  every=every, t0=t_c0)
        t_cg = time.perf_counter() - t_c0

    b_norm = float(np.linalg.norm(rhs))
    resid_pre = float(np.linalg.norm(rhs - A @ x) / max(b_norm, 1e-300))
    n_ref, q_pre, t_ref = 0, None, 0.0
    if M is not None:
        n_ref = int(p["refine"] if refine is None else refine)
        if n_ref > 0:                     # 回代前的 Q, 用作迭代误差自检
            Tp = np.asarray(x, dtype=np.float64).reshape(nz, ny, nx)
            q_pre = float(asm["Rx"] * np.sum(asm["Tb"] - Tp[:, :, 0]))
        t_r0 = time.perf_counter()
        for _ in range(n_ref):
            x = x + M @ (rhs - A @ x)
        t_ref = time.perf_counter() - t_r0
    wall = time.perf_counter() - t0
    resid = float(np.linalg.norm(rhs - A @ x) / max(b_norm, 1e-300))

    t_s0 = time.perf_counter()                        # T-2 第四个计时点: 后处理
    T = np.asarray(x, dtype=np.float64).reshape(nz, ny, nx)
    Hx, Hy, Hz, Rx = asm["Hx"], asm["Hy"], asm["Hz"], asm["Rx"]
    Tb, Tinf, W = asm["Tb"], asm["Tinf"], asm["W"]
    fl = dict(root=float(Rx * np.sum(Tb - T[:, :, 0])),
              tip=float(Hx * np.sum(T[:, :, -1] - Tinf)),
              y0=float(Hy * np.sum(T[:, 0, :] - Tinf)),
              y1=float(Hy * np.sum(T[:, -1, :] - Tinf)))
    if asm["z_conv"]:
        fl["z0"] = float(Hz * np.sum(T[0, :, :] - Tinf))
        fl["z1"] = float(Hz * np.sum(T[-1, :, :] - Tinf))
    else:
        fl["z0"] = fl["z1"] = 0.0
    q_in = fl["root"]
    q_out = fl["tip"] + fl["y0"] + fl["y1"] + fl["z0"] + fl["z1"]
    dq = q_in - q_out
    t_post = time.perf_counter() - t_s0

    out = dict(Q=q_in, Q_per_span=q_in / W, Q_out=q_out, Q_pre=q_pre,
               abs_diff=abs(dq), imbalance=dq / q_in,
               T=T, fluxes=fl,
               grid=dict(nx=nx, ny=ny, nz=nz, dx=asm["dx"], dy=asm["dy"],
                         dz=asm["dz"], ar_xy=asm["dx"] / asm["dy"],
                         ar_zy=asm["dz"] / asm["dy"], N=n, nnz=asm["nnz"],
                         csr_mb=asm["csr_mb"],
                         z_bc=("h" if asm["z_conv"] else "adiabatic"),
                         A_total=2.0 * asm["L"] * W + 2.0 * asm["L"] * asm["t"]
                         + asm["t"] * W),
               solve=dict(method=used, precond=precond, iters=n_iter,
                          info=int(info), converged=(info == 0),
                          wall_s=wall, residual=resid, resid_pre=resid_pre,
                          refine=n_ref, rtol=rtol,
                          t_asm=t_asm, t_prec=t_prec, t_cg=t_cg, t_ref=t_ref,
                          t_post=t_post, t_total=t_asm + wall + t_post, fnz=fnz),
               Tmax=float(T.max()), Tmin=float(T.min()))
    if verbose:
        print("[fin3d] %dx%dx%d N=%d nnz=%d %s iters=%d wall=%.2fs resid=%.3e Q=%.6f"
              % (nx, ny, nz, n, asm["nnz"], used, n_iter, wall, resid, q_in))
    return out


def analytic_fin(params, with_tip=True):
    """一维肋片解析解 (单位展向, W/m).

    with_tip=True  : 端面对流修正 beta = h/(m*k)
    with_tip=False : 绝热端 (beta = 0)
    """
    p = dict(DEFAULTS)
    p.update({k: v for k, v in dict(params).items() if v is not None})
    L, t = float(p["L"]), float(p["t"])
    kc, h = float(p["k"]), float(p["h"])
    tb = float(p["Tb"]) - float(p["Tinf"])
    m = np.sqrt(2.0 * h / (kc * t))
    beta = (h / (m * kc)) if with_tip else 0.0
    mL = m * L
    den = np.cosh(mL) + beta * np.sinh(mL)
    q = tb * np.sqrt(h * 2.0 * kc * t) * (np.sinh(mL) + beta * np.cosh(mL)) / den
    return float(q)
