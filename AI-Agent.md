# AI 工具使用记录

本项目允许使用 AI 工具，但关键模块、数值假设和最终报告均经过人工检查与运行验证。

## Codex / GPT-5

### 关键提示词摘要

- “首先仔细理解整个项目文件，然后阅读 PROJECT_SPEC.pdf。”
- “严格按照 project_spec.pdf 每一个要求完成这份作业。”
- “最后对照检查格式要求、必做项、选做加分项、致谢、交付物是否全部符合要求。”

### 主要产出

- 理解 `PROJECT_SPEC.pdf`，整理 M1--M8 的交付要求。
- 协助设计 Python 模块结构：`conics.py`, `nbody.py`, `swingby.py`, `mission.py`, `plots.py`, `run_all.py`。
- 协助实现 Velocity-Verlet 积分器、拼接圆锥曲线公式、月球借力解析公式和全年扫描脚本。
- 协助生成 LaTeX 报告、README、Makefile 和本 AI 使用记录。
- 协助根据评分反馈补充选做项：
  - `extensions.py` 的 3D 月球倾角窗口分析与 Schwarzschild 相对论修正；
  - `lambert.py` 的通用变量 Lambert 求解器；
  - `multiflyby.py` 的 Earth → Moon → Venus → Sun → Earth 多次借力残差预算，并补充 Venus Horizons 离线缓存；
  - `interactive_demo.py` 的 Tkinter 交互演示和无显示自测；
  - `surrogate_design.py` 的机器学习辅助轨道筛选、交叉验证和置换重要性分析。
- 协助根据助教修正规则补充真实闭合主任务：
  - `real_closure.py` 使用真实 Horizons 月球交会与真实返回地球状态做 Lambert 拼接；
  - 将月球双曲线被动偏折无法闭合的速度矢量差计入 `lunar_residual_delta_v_km_s`；
  - 生成 `mission_summary.json`、`today_solution.json` 与带时间戳轨迹 CSV，供规则 13--17 机器复核；
  - 增加 `data/horizons_cache_2026_2029.json`，保证 2026 年发射后两年内返回窗口可离线复核。
- 协助调试：
  - 修正 Horizons 对照中的太阳质心参考系转换；
  - 调整 M1 基准表格中半通径的有效数字比较；
  - 检查 `make all` 的构建流程。

### 人工修正与确认

- 保留课程提供的离线 Horizons 缓存作为唯一外部数据源，避免网络依赖。
- 对所有关键输出运行 `python src/run_all.py` 进行验证。
- 在报告中明确说明半解析月球借力模型的假设、适用范围和误差来源。
- 对照 `PROJECT_SPEC.pdf` 修正 README 最优近月距、报告 M1--M8 小节标题、O5 选做项说明、`Makefile clean` 清理范围。
- 对照评分反馈逐项复查规则 22--27，运行 `python src/run_all.py`、`python src/self_check.py`、`python src/interactive_demo.py --self-test` 和 LaTeX 编译，确认报告数值与生成的 JSON/CSV 一致。
- 对照修正规则复查真实月相、真实地球返回和速度预算口径，运行 `python src/run_all.py`、`python src/self_check.py`、`python -m py_compile`、XeLaTeX 编译，并人工抽查 `mission_summary.json` 与轨迹 CSV。
