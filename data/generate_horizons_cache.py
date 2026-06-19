#!/usr/bin/env python3
"""
generate_horizons_cache.py

从 JPL Horizons 获取 2026 全年（365 天，每日一个历元）
Sun、Earth、Moon 在太阳质心系（J2000 黄道面）下的三维状态向量，
保存到 horizons_cache_2026.json 供离线使用。

依赖：numpy, astroquery, astropy
运行：conda run -n Teaching python generate_horizons_cache.py
"""

import sys
import json
from pathlib import Path

import numpy as np
from astroquery.jplhorizons import Horizons

# ---------- 代理补丁（必须在任何 Horizons 调用之前导入） ----------
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "week9" / "src"))
import jpl_forward  # noqa: F401

# ---------- 常数 ----------
AU_KM   = 149_597_870.7     # 1 AU = ? km
DAY_SEC = 86_400.0           # 1 day = ? s

START = "2026-01-01"
STOP  = "2027-01-01"
STEP  = "1d"
CENTER = "@10"               # 太阳质心

BODIES = [
    {"id": "10",  "name": "Sun"},
    {"id": "399", "name": "Earth"},
    {"id": "301", "name": "Moon"},
]

OUT = Path(__file__).resolve().parent / "horizons_cache_2026.json"


def fetch(body_id: str, name: str) -> list[dict]:
    """获取单天体全年状态向量，返回 list[dict]。"""
    print(f"  正在获取 {name} (id={body_id}) ...")
    obj = Horizons(
        id=body_id,
        location=CENTER,
        epochs={"start": START, "stop": STOP, "step": STEP},
    )
    tbl = obj.vectors(refplane="ecliptic", aberrations="geometric")

    n = len(tbl)
    records = []
    for i in range(n):
        records.append({
            "jd_tdb":       float(tbl["datetime_jd"][i]),
            "calendar":     str(tbl["datetime_str"][i]),
            # 位置：AU → km
            "x_km":         float(tbl["x"][i])  * AU_KM,
            "y_km":         float(tbl["y"][i])  * AU_KM,
            "z_km":         float(tbl["z"][i])  * AU_KM,
            # 速度：AU/day → km/s
            "vx_km_s":      float(tbl["vx"][i]) * AU_KM / DAY_SEC,
            "vy_km_s":      float(tbl["vy"][i]) * AU_KM / DAY_SEC,
            "vz_km_s":      float(tbl["vz"][i]) * AU_KM / DAY_SEC,
        })
    print(f"    获取 {n} 个历元")
    return records


def main():
    cache = {
        "description": "Sun-Earth-Moon 状态向量缓存，2026 全年每日一个历元",
        "center":       "Sun body center (@10)",
        "refplane":     "J2000 ecliptic",
        "units":        {"position": "km", "velocity": "km/s"},
        "start":        START,
        "stop":         STOP,
        "step":         STEP,
        "bodies":       {},
    }

    for body in BODIES:
        cache["bodies"][body["name"]] = {
            "horizons_id": body["id"],
            "epochs":      fetch(body["id"], body["name"]),
        }

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)

    size_mb = OUT.stat().st_size / 1024 / 1024
    print(f"\n已写入 {OUT}")
    print(f"文件大小: {size_mb:.2f} MB")

    # ---------- 快速校验 ----------
    earth = cache["bodies"]["Earth"]["epochs"][0]
    r = np.sqrt(earth["x_km"]**2 + earth["y_km"]**2 + earth["z_km"]**2)
    v = np.sqrt(earth["vx_km_s"]**2 + earth["vy_km_s"]**2 + earth["vz_km_s"]**2)
    print(f"\n[校验] Earth @ {earth['calendar']}")
    print(f"  |r| = {r:.3e} km  (期望 ~1.5e8 km = 1 AU)")
    print(f"  |v| = {v:.3f} km/s (期望 ~29-30 km/s)")

    sun0 = cache["bodies"]["Sun"]["epochs"][0]
    print(f"\n[校验] Sun @ {sun0['calendar']}")
    print(f"  x,y,z = {sun0['x_km']:.6f}, {sun0['y_km']:.6f}, {sun0['z_km']:.6f} km")


if __name__ == "__main__":
    main()
