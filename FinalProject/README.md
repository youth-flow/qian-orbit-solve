# 钱学森问题扩展求解

本仓库为《人工智能与数学软件》期末设计项目，实现 `PROJECT_SPEC.pdf` 要求的月球借力与发射窗口优化版本。

## 环境

- Python 3.10+
- Python 包：`numpy`, `matplotlib`, `imageio`, `imageio-ffmpeg`
- XeLaTeX：用于编译中文报告

本实现不依赖 `scipy`，优化与验证均用 `numpy` 完成。
仓库还包含自实现通用变量 Lambert 求解器 `src/lambert.py`，用于选做项和自检，不调用 `scipy.optimize`。

## 5 分钟复现

```powershell
make all
```

该命令会：

1. 读取 `data/horizons_cache_2026.json` 与 `data/horizons_cache_2026_2029.json`；
2. 运行 `src/run_all.py`，生成验证结果、扫描数据、机器可读 JSON/CSV、PNG 图表和 MP4 动画；
3. 用 XeLaTeX 编译 `report.tex` 为 `report.pdf`。

若只想重新生成数值与图表：

```powershell
make data
```

若只想重新编译报告：

```powershell
make pdf
```

清理生成文件：

```powershell
make clean
```


## 目录

- `src/`：所有源代码。
- `data/horizons_cache_2026.json`：课程提供的 Sun-Earth-Moon 离线 Horizons 缓存。
- `data/horizons_cache_2026_2029.json`：真实返回闭合复核使用的 Sun-Earth-Moon Horizons 缓存，查询中心为 `CENTER='@10'`。
- `data/venus_horizons_cache_2026.json`：O4 多次借力探索使用的 Venus 日心 Horizons 缓存，查询中心为 `CENTER='@10'`。
- `data/generated/`：由 `make data` 生成的 JSON、CSV、PNG 和 MP4，不提交。
- `report.tex`：报告源文件。
- `AI-Agent.md`：AI 工具使用记录。

## 机器可读输出

`make data` 会生成以下复核文件：

- `data/generated/results.json`：M1--M8 汇总结果。
- `data/generated/grading_check.json`：面向评分规则的自动自检摘要。
- `data/generated/mission_summary.json`：全年最优真实闭合解摘要，包含发射、真实月球交会、真实地球返回、近月距、近日距、三段速度增量和节能比例。
- `data/generated/today_solution.json`：以 2026-06-19 为输入日期的同口径解。
- `data/generated/mission_trajectory_day223.csv`、`today_trajectory_day169.csv`：带时间戳的轨道状态序列。
- `data/generated/mission_events_day223.csv`、`today_events_day169.csv`：关键事件表。
- `data/generated/scan_daily_best.csv`：365 个发射日逐日最优结果。
- `data/generated/legacy_analytic_scan.json`：旧解析筛选模型结果，仅作对照，不作为主任务真实性复核依据。
- `data/generated/multi_flyby_summary.json`、`multi_flyby_candidates.csv`：Earth → Moon → Venus → Sun → Earth 多次借力探索与残差预算。
- `data/generated/surrogate_design_summary.json`、`surrogate_verified_candidates.csv`、`surrogate_permutation_importance.csv`：机器学习辅助轨道筛选扩展。

JPL Horizons 在线接口位于 `src/jpl_access.py`，代理和 astroquery 参数均使用 `CENTER='@10'`。
`results.json` 还包含 3D 月球倾角影响、广义相对论近日点修正、Lambert 求解器验证、多次借力探索、Tkinter 交互演示自测和机器学习辅助筛选结果，对应代码位于 `src/extensions.py`、`src/lambert.py`、`src/multiflyby.py`、`src/interactive_demo.py` 与 `src/surrogate_design.py`。

交互演示可运行：

```powershell
python src/interactive_demo.py
```

无显示环境下可先验证交互逻辑：

```powershell
python src/interactive_demo.py --self-test
```

在 Xvfb 或本地桌面环境中可运行自动 GUI 冒烟测试：

```powershell
python src/interactive_demo.py --smoke-gui
```

生成结果后也可以单独运行：

```powershell
python src/self_check.py
```

该命令会重新读取 `data/generated/` 下的 JSON/CSV 并输出规则映射检查。

## 主要结果

当前模型扫描 2026 年 365 个发射日期，得到最优窗口：

- 发射日期：2026-08-12
- 月球交会：2026-08-15
- 返回地球：2028-08-02 22:12 UTC
- 近月距：1838.0 km
- 日心近日距：0.248 AU
- 总速度增量：约 20.099 km/s

主结果由 `src/real_closure.py` 生成：发射点、月球交会点和返回地球点均取对应时刻的真实 Horizons 状态；月球双曲线被动偏折不能闭合的速度矢量差完整计入 `lunar_residual_delta_v_km_s`。旧的 13.214 km/s 解析筛选结果仍保存在 `legacy_analytic_scan.json` 中，用于说明“不检查真实月相/真实返回”会低估代价。

详细数值以 `make data` 后生成的 `data/generated/results.json` 为准。
