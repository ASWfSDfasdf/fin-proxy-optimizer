# fin-proxy-optimizer

> 二维单肋稳态导热的数值解 + 参数扫描 + 代理模型寻优。
> 面向**工业仿真**场景：给定一块肋片（散热器翅片）的几何与换热条件，
> 算它的散热能力，再在参数空间里找最优。

## 一句话

**给 AI agent 用的一句话：** 「把这份翅片参数跑一遍扫描，拟合代理模型，告诉我散热最好的尺寸组合，并给出误差。」

---

## 1. 能做什么

| 阶段 | 命令 | 产出 |
|---|---|---|
| 自检 | `python fin2d.py --check` | 与一维解析解对比，误差 < 1% 才算通过 |
| 单算例 | `python fin2d.py --case case.json` | 温度分布 `profile.csv` + 汇总 JSON |
| 参数扫描 | `python fin2d.py --scan scan.json --outdir out` | 每个算例一份 profile + 汇总 `results.csv` |
| 代理模型 | `python fit_surrogate.py --csv out/results.csv` | LOO 精度 + 最优参数点 |

**不需要 GPU，不需要编译，不需要联网。** 只要 numpy（scipy 可选）。

---

## 2. 快速开始

```bash
python fin2d.py --check          # 先自检，确认环境正常
python fin2d.py --scan scan_demo.json --outdir demo_out
python fit_surrogate.py --csv demo_out/results.csv --target Q_W_per_m
python fit_surrogate.py --csv demo_out/results.csv --target Q_per_vol
```

`scan_demo.json`：

```json
{
  "base": {"k": 200.0, "tb": 350.0, "tinf": 300.0, "nx": 79, "ny": 33},
  "grid": {
    "L": [0.03, 0.05, 0.07],
    "t": [0.001, 0.002, 0.003],
    "h": [30.0, 60.0, 90.0]
  }
}
```

---

## 3. 物理与数值

**控制方程**：肋片内的二维稳态导热（沿长度 x、厚度 y）

```
∂²T/∂x² + ∂²T/∂y² = 0
```

**边界条件**

| 位置 | 条件 | 类型 |
|---|---|---|
| x = 0（根部） | T = tb | 第一类 |
| x = L（端部） | −k ∂T/∂x = h(T − t∞) | 第三类 |
| y = 0, t（两侧） | ∓k ∂T/∂y = h(T − t∞) | 第三类 |

**离散**：均匀网格五点差分；边界用**半格控制体**，把边界的半格导热一并计入
（这一步不做会导致热流一阶误差，实测差 20%）。求解用 scipy 稀疏 LU；
无 scipy 时小网格退回 numpy 稠密解。

**热流**：根部 `Q = −k ∫ (∂T/∂x)|₀ dy`，导数用二阶单侧差分、
沿厚度用**梯形积分**（矩形积分会多算一格，实测偏大 3%）。

---

## 4. 自检（为什么可信）

`--check` 会做两件事：

1. **网格加密收敛**：ny = 11→33（有 scipy 时到 81），温度误差应单调下降；
2. **与一维解析解对比**：薄肋极限下二维解应趋于

```
θ(x)/θb = [cosh(m(L−x)) + β sinh(m(L−x))] / [cosh(mL) + β sinh(mL)]
m = √(2h/(k·t))，β = h/(m·k)，θ = T − t∞
```

实测（默认算例，Bi = 5×10⁻⁴）：

```
   ny    Q_numeric   Q_analytic     Q_err   T_err_max
   11     210.6658     211.1104    0.211%      0.163%
   17     210.6708     211.1104    0.208%      0.159%
   25     210.6734     211.1104    0.207%      0.158%
   33     210.6880     211.1104    0.200%      0.164%
自检结果： OK
```

剩余 0.2% 是**二维与一维的真实差异**（厚度方向导热），不是离散误差——
所以它对网格不敏感，这正是我们想证明的东西。

---

## 5. 参数

见 `params.schema.json`。要点：

- `L` 肋高、`t` 肋厚（m）；`k` 导热系数、`h` 换热系数
- `tb` 根部温度、`tinf` 环境温度（K）
- `nx`/`ny` 网格；自检保证收敛，一般 `nx=201, ny=41` 足够

**适用区间**：薄肋，`Bi = h·t/k < 0.1`。超过这个范围一维解析解不再适用
（二维解仍然正确，但 `--check` 的阈值要放宽）。

---

## 6. 代理模型与寻优

`fit_surrogate.py` 用**高斯 RBF 插值**拟合「参数 → 目标」，
用**留一交叉验证（LOO）**给出泛化误差，再在参数盒内做分块网格搜索。

实测（27 个算例）：

| 目标 | LOO R² | MAPE | 最优点 |
|---|---|---|---|
| `Q_W_per_m`（绝对散热） | 0.9976 | 1.47% | L=0.07, t=0.003, h=90 |
| `Q_per_vol`（单位体积散热） | 0.9848 | 4.78% | L=0.03, t=0.001, h=90 |

**两个目标给出相反的答案**——这就是代理模型的价值：真求解器只告诉你
「这个点多少」，代理模型让你能在一秒内看完整片参数空间。

> ⚠ 单目标 `Q_W_per_m` 是单调的（越长越厚越强），最优点总在盒角落。
> 真正有意思的是**带约束的目标**（如 `Q_per_vol`、或「Q 最大且质量不超过 m」）。

---

## 7. 已知边界

- **只算单肋**，不算整片散热器（多肋间距、旁通流、辐射都没有）
- `h` 是**给定值**，不做流固耦合（不预测风速→h 的关系）
- 二维（无限展向），三维效应（端部效应、肋间流动）不在模型内
- 不做瞬态、不做变物性（`k`、`h` 视为常数）

---

## 8. 扩展路线

1. **三维单肋**（展向有限长）——看端部效应
2. **多肋阵列**——肋间距 s 进入参数，出现「间距太大浪费空间、太小风阻大」的真实折中
3. **h 的关联式**——把风速 u 变成输入，用 Nu = f(Re, Pr) 估 h，接上真实工况
4. **与 OpenFOAM 对齐**——用 `module load OpenFOAM/9-openmpi415-gcc-9.3.0` 算同样工况做交叉验证

---

## 9. 在超算上跑

```bash
# 家目录 /work/home/<账号>
module load anaconda3/2023.09        # Python 3.11.5 + numpy + scipy
python fin2d.py --check

# 批量扫描放进作业脚本（推荐 8~16 核、单作业 2 小时内）
sbatch run_scan.sbatch
```

`run_scan.sbatch` 用绝对路径解释器，避免批量作业里 module 不可用：

```bash
#!/bin/bash
#SBATCH --partition=xahctest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

cd "$SLURM_SUBMIT_DIR"
/public/software/apps/anaconda3/2023.09/bin/python3 fin2d.py \
    --scan scan_demo.json --outdir "out_${SLURM_JOB_ID}"
```

---

## 10. 目录

```
fin2d.py            二维求解器（自检/单算例/扫描）
fit_surrogate.py    RBF 代理模型 + LOO + 盒内寻优
params.schema.json  参数契约（给 agent 读的机器可读定义）
scan_demo.json      示例扫描配置
SKILL.md            给 agent 的技能说明（本文件之外的调用约定）
icon.png            256×256 图标
```

## License

MIT
