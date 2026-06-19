# qian-orbit-solve

钱学森《星际航行概论》相关期末项目整理仓库。可复现主项目位于 `FinalProject/`，根目录 `Makefile` 已代理到该目录。

## 运行

```powershell
make all
```

该命令会进入 `FinalProject/`，运行数值计算、生成 JSON/CSV/PNG/MP4，并用 XeLaTeX 编译 `FinalProject/report.pdf`。

也可以只运行：

```powershell
make data
make pdf
make clean
```

## 目录

- `FinalProject/`：最终提交项目，包含 `report.tex`、`Makefile`、`README.md`、`AI-Agent.md`、`src/` 和离线 Horizons 缓存。
- `data/`：早期课程提供/生成缓存脚本与 2026 Sun-Earth-Moon 离线缓存。
- `2D_gravity/`：二维引力演示与记录。
- `report.tex`、`report.pdf`：根目录早期报告材料，最终评分以 `FinalProject/` 为准。

## 公开仓库清理

未纳入公开 GitHub 仓库的本地文件包括：

- `JPL_API.env`：本地 JPL 代理密钥。
- `星际航行概论-钱学森-2008年版.pdf`、`行星航行概论-钱学森-1963年版.pdf`：大体积版权书籍 PDF。
- 题面 `PROJECT_SPEC.tex/pdf`：本地版本包含课程代理凭据，不适合公开发布。
- `FinalProject/data/generated/`：可由 `make all` 重新生成。

在线 JPL 代理查询代码位于 `FinalProject/src/jpl_access.py`。公开版本不内置凭据；如需在线查询，请参考 `JPL_API.example.env` 自行设置 `JPL_API` 和 `JPL_TOKEN`。默认构建使用仓库内离线 Horizons 缓存，不需要在线密钥。
