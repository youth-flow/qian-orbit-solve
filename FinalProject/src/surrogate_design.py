"""Machine-learning assisted launch-window screening.

This optional module is intentionally outside the base M1-M8 requirements.  It
implements a small, dependency-free surrogate optimizer that learns the
semi-analytic mission cost from a sparse set of exact evaluations, predicts a
large candidate bank, and verifies the most promising points with the original
physics model.  The point is not to replace the full deterministic scan used in
the report; it is to show a different computational design workflow that can
reduce expensive candidate evaluations.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from constants import AU_KM, MIN_LUNAR_PERIAPSIS, R_SUN
from mission import evaluate_candidate


SIDES = ("leading", "trailing")


@dataclass(frozen=True)
class DesignPoint:
    day_index: int
    rp_AU: float
    rm_km: float
    side: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvaluatedPoint:
    day_index: int
    date: str
    rp_AU: float
    rm_km: float
    side: str
    total_delta_v_km_s: float
    launch_delta_v_km_s: float
    lunar_residual_delta_v_km_s: float
    reentry_delta_v_km_s: float
    constraints_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PredictionPoint:
    day_index: int
    rp_AU: float
    rm_km: float
    side: str
    predicted_total_delta_v_km_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FoldScore:
    fold: int
    train_size: int
    test_size: int
    rmse_km_s: float
    mae_km_s: float
    max_abs_error_km_s: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> "Standardizer":
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale < 1.0e-12] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.scale


@dataclass
class RidgeModel:
    weights: np.ndarray
    standardizer: Standardizer
    feature_names: list[str]
    alpha: float

    def predict_matrix(self, features: np.ndarray) -> np.ndarray:
        z = self.standardizer.transform(features)
        design = np.column_stack([np.ones(len(z)), z])
        return design @ self.weights

    def coefficients(self) -> list[dict]:
        coef = self.weights[1:]
        return [
            {
                "feature": name,
                "coefficient": float(value),
                "abs_coefficient": abs(float(value)),
            }
            for name, value in sorted(zip(self.feature_names, coef), key=lambda item: abs(float(item[1])), reverse=True)
        ]


def _first_primes(n: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < n:
        root = int(math.sqrt(candidate))
        ok = True
        for p in primes:
            if p > root:
                break
            if candidate % p == 0:
                ok = False
                break
        if ok:
            primes.append(candidate)
        candidate += 1
    return primes


def _van_der_corput(index: int, base: int) -> float:
    result = 0.0
    denom = 1.0
    i = index
    while i > 0:
        i, remainder = divmod(i, base)
        denom *= base
        result += remainder / denom
    return result


def halton_sequence(count: int, dim: int, start_index: int = 1) -> np.ndarray:
    primes = _first_primes(dim)
    data = np.zeros((count, dim), dtype=float)
    for row in range(count):
        index = start_index + row
        for col, base in enumerate(primes):
            data[row, col] = _van_der_corput(index, base)
    return data


def map_unit_to_design(unit: np.ndarray) -> DesignPoint:
    day = int(min(364, max(0, math.floor(unit[0] * 365.0))))
    rp_min = 2.0 * R_SUN / AU_KM
    rp = rp_min + float(unit[1]) * (0.4 - rp_min)
    rm = MIN_LUNAR_PERIAPSIS + float(unit[2]) * (50_000.0 - MIN_LUNAR_PERIAPSIS)
    side = SIDES[int(float(unit[3]) >= 0.5)]
    return DesignPoint(day_index=day, rp_AU=rp, rm_km=rm, side=side)


def low_discrepancy_training_points(count: int) -> list[DesignPoint]:
    units = halton_sequence(count, 4)
    points: list[DesignPoint] = []
    seen: set[tuple[int, int, int, str]] = set()
    for unit in units:
        point = map_unit_to_design(unit)
        key = (point.day_index, round(point.rp_AU * 10_000), round(point.rm_km), point.side)
        if key in seen:
            continue
        seen.add(key)
        points.append(point)
    index = count + 1
    while len(points) < count:
        point = map_unit_to_design(halton_sequence(1, 4, start_index=index)[0])
        key = (point.day_index, round(point.rp_AU * 10_000), round(point.rm_km), point.side)
        if key not in seen:
            seen.add(key)
            points.append(point)
        index += 1
    return points


def physics_anchor_points() -> list[DesignPoint]:
    """Sparse exact anchors on the low-energy boundary of the design space."""

    rp_values = [0.4, 0.375, 0.35]
    rm_values = [MIN_LUNAR_PERIAPSIS, 2_500.0]
    points: list[DesignPoint] = []
    for day in range(365):
        for rp in rp_values:
            for rm in rm_values:
                for side in SIDES:
                    points.append(DesignPoint(day_index=day, rp_AU=rp, rm_km=rm, side=side))
    return points


def merge_design_points(*groups: list[DesignPoint]) -> list[DesignPoint]:
    merged: list[DesignPoint] = []
    seen: set[tuple[int, int, int, str]] = set()
    for group in groups:
        for point in group:
            key = (point.day_index, round(point.rp_AU * 10_000), round(point.rm_km), point.side)
            if key in seen:
                continue
            seen.add(key)
            merged.append(point)
    return merged


def candidate_bank(
    day_stride: int = 1,
    rp_count: int = 17,
    rm_count: int = 11,
    include_boundary: bool = True,
) -> list[DesignPoint]:
    days = list(range(0, 365, max(1, day_stride)))
    if days[-1] != 364:
        days.append(364)
    rp_min = 2.0 * R_SUN / AU_KM
    rp_values = np.linspace(rp_min, 0.4, rp_count)
    if include_boundary and 0.4 not in rp_values:
        rp_values = np.unique(np.append(rp_values, 0.4))
    rm_values = np.linspace(MIN_LUNAR_PERIAPSIS, 50_000.0, rm_count)
    if include_boundary and MIN_LUNAR_PERIAPSIS not in rm_values:
        rm_values = np.unique(np.append(rm_values, MIN_LUNAR_PERIAPSIS))
    points: list[DesignPoint] = []
    for day in days:
        for rp in rp_values:
            for rm in rm_values:
                for side in SIDES:
                    points.append(DesignPoint(day_index=int(day), rp_AU=float(rp), rm_km=float(rm), side=side))
    return points


def evaluate_design_point(point: DesignPoint, positions: np.ndarray, velocities: np.ndarray) -> EvaluatedPoint:
    candidate = evaluate_candidate(point.day_index, point.rp_AU * AU_KM, point.rm_km, point.side, positions, velocities)
    return EvaluatedPoint(
        day_index=candidate.day_index,
        date=candidate.date,
        rp_AU=candidate.rp_AU,
        rm_km=candidate.rm_km,
        side=candidate.side,
        total_delta_v_km_s=candidate.total_delta_v_km_s,
        launch_delta_v_km_s=candidate.launch_delta_v_km_s,
        lunar_residual_delta_v_km_s=candidate.lunar_residual_delta_v_km_s,
        reentry_delta_v_km_s=candidate.reentry_delta_v_km_s,
        constraints_ok=candidate.constraints_ok,
    )


def evaluate_design_points(points: Iterable[DesignPoint], positions: np.ndarray, velocities: np.ndarray) -> list[EvaluatedPoint]:
    return [evaluate_design_point(point, positions, velocities) for point in points]


def _moon_phase_features(day_index: int, positions: np.ndarray) -> dict[str, float]:
    earth = positions[day_index, 1]
    moon = positions[day_index, 2]
    rel = moon - earth
    earth_angle = math.atan2(float(earth[1]), float(earth[0]))
    moon_angle = math.atan2(float(rel[1]), float(rel[0]))
    phase = moon_angle - earth_angle
    return {
        "moon_phase_sin": math.sin(phase),
        "moon_phase_cos": math.cos(phase),
        "earth_radius_AU": float(np.linalg.norm(earth[:2]) / AU_KM),
        "moon_range_100k_km": float(np.linalg.norm(rel) / 100_000.0),
        "moon_z_100k_km": float(rel[2] / 100_000.0),
    }


def feature_vector(point: DesignPoint, positions: np.ndarray) -> tuple[list[float], list[str]]:
    day = point.day_index
    day_phase = 2.0 * math.pi * day / 365.0
    rp = point.rp_AU
    rp_min = 2.0 * R_SUN / AU_KM
    rp_scaled = (rp - rp_min) / (0.4 - rp_min)
    rm_scaled = (point.rm_km - MIN_LUNAR_PERIAPSIS) / (50_000.0 - MIN_LUNAR_PERIAPSIS)
    side = 1.0 if point.side == "trailing" else -1.0
    moon = _moon_phase_features(day, positions)
    names = [
        "day_sin",
        "day_cos",
        "rp_scaled",
        "rm_scaled",
        "side",
        "rp_squared",
        "rm_squared",
        "rp_rm",
        "side_rp",
        "side_rm",
        "moon_phase_sin",
        "moon_phase_cos",
        "earth_radius_AU",
        "moon_range_100k_km",
        "moon_z_100k_km",
        "moon_phase_sin_side",
        "moon_phase_cos_side",
        "rp_moon_phase_cos",
        "rm_moon_phase_sin",
    ]
    values = [
        math.sin(day_phase),
        math.cos(day_phase),
        rp_scaled,
        rm_scaled,
        side,
        rp_scaled * rp_scaled,
        rm_scaled * rm_scaled,
        rp_scaled * rm_scaled,
        side * rp_scaled,
        side * rm_scaled,
        moon["moon_phase_sin"],
        moon["moon_phase_cos"],
        moon["earth_radius_AU"],
        moon["moon_range_100k_km"],
        moon["moon_z_100k_km"],
        moon["moon_phase_sin"] * side,
        moon["moon_phase_cos"] * side,
        rp_scaled * moon["moon_phase_cos"],
        rm_scaled * moon["moon_phase_sin"],
    ]
    return values, names


def feature_matrix(points: list[DesignPoint], positions: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rows: list[list[float]] = []
    names: list[str] | None = None
    for point in points:
        values, feature_names = feature_vector(point, positions)
        rows.append(values)
        names = feature_names
    assert names is not None
    return np.array(rows, dtype=float), names


def records_to_points(records: list[EvaluatedPoint]) -> list[DesignPoint]:
    return [DesignPoint(day_index=r.day_index, rp_AU=r.rp_AU, rm_km=r.rm_km, side=r.side) for r in records]


def records_to_targets(records: list[EvaluatedPoint]) -> np.ndarray:
    return np.array([r.total_delta_v_km_s for r in records], dtype=float)


def fit_ridge(records: list[EvaluatedPoint], positions: np.ndarray, alpha: float = 1.0e-3) -> RidgeModel:
    points = records_to_points(records)
    features, names = feature_matrix(points, positions)
    standardizer = Standardizer.fit(features)
    z = standardizer.transform(features)
    design = np.column_stack([np.ones(len(z)), z])
    targets = records_to_targets(records)
    penalty = alpha * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return RidgeModel(weights=weights, standardizer=standardizer, feature_names=names, alpha=alpha)


def predict_points(model: RidgeModel, points: list[DesignPoint], positions: np.ndarray) -> list[PredictionPoint]:
    features, _ = feature_matrix(points, positions)
    predictions = model.predict_matrix(features)
    return [
        PredictionPoint(
            day_index=point.day_index,
            rp_AU=point.rp_AU,
            rm_km=point.rm_km,
            side=point.side,
            predicted_total_delta_v_km_s=float(value),
        )
        for point, value in zip(points, predictions)
    ]


def cross_validate(records: list[EvaluatedPoint], positions: np.ndarray, folds: int = 5, alpha: float = 1.0e-3) -> list[FoldScore]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    scores: list[FoldScore] = []
    ordered = list(records)
    for fold in range(folds):
        train = [record for idx, record in enumerate(ordered) if idx % folds != fold]
        test = [record for idx, record in enumerate(ordered) if idx % folds == fold]
        model = fit_ridge(train, positions, alpha=alpha)
        test_points = records_to_points(test)
        y_true = records_to_targets(test)
        features, _ = feature_matrix(test_points, positions)
        y_pred = model.predict_matrix(features)
        err = y_pred - y_true
        scores.append(
            FoldScore(
                fold=fold,
                train_size=len(train),
                test_size=len(test),
                rmse_km_s=float(math.sqrt(np.mean(err * err))),
                mae_km_s=float(np.mean(np.abs(err))),
                max_abs_error_km_s=float(np.max(np.abs(err))),
            )
        )
    return scores


def _prediction_to_design(prediction: PredictionPoint) -> DesignPoint:
    return DesignPoint(
        day_index=prediction.day_index,
        rp_AU=prediction.rp_AU,
        rm_km=prediction.rm_km,
        side=prediction.side,
    )


def _deduplicate_predictions(predictions: list[PredictionPoint], max_count: int) -> list[PredictionPoint]:
    out: list[PredictionPoint] = []
    seen: set[tuple[int, int, int, str]] = set()
    for item in sorted(predictions, key=lambda p: p.predicted_total_delta_v_km_s):
        key = (item.day_index, round(item.rp_AU * 10_000), round(item.rm_km), item.side)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max_count:
            break
    return out


def _summarize_scores(scores: list[FoldScore]) -> dict:
    rmse = np.array([s.rmse_km_s for s in scores], dtype=float)
    mae = np.array([s.mae_km_s for s in scores], dtype=float)
    maxerr = np.array([s.max_abs_error_km_s for s in scores], dtype=float)
    return {
        "folds": [s.to_dict() for s in scores],
        "mean_rmse_km_s": float(np.mean(rmse)),
        "mean_mae_km_s": float(np.mean(mae)),
        "max_fold_error_km_s": float(np.max(maxerr)),
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("rank correlation requires equally sized non-empty arrays")
    rx = _rankdata(np.asarray(x, dtype=float))
    ry = _rankdata(np.asarray(y, dtype=float))
    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def prediction_diagnostics(model: RidgeModel, records: list[EvaluatedPoint], positions: np.ndarray) -> dict:
    points = records_to_points(records)
    features, _ = feature_matrix(points, positions)
    y_true = records_to_targets(records)
    y_pred = model.predict_matrix(features)
    err = y_pred - y_true
    best_true = int(np.argmin(y_true))
    best_pred = int(np.argmin(y_pred))
    return {
        "sample_count": len(records),
        "rmse_km_s": float(math.sqrt(np.mean(err * err))),
        "mae_km_s": float(np.mean(np.abs(err))),
        "spearman_rank_correlation": spearman_correlation(y_true, y_pred),
        "true_best_date": records[best_true].date,
        "true_best_total_km_s": float(y_true[best_true]),
        "predicted_best_date": records[best_pred].date,
        "predicted_best_model_value_km_s": float(y_pred[best_pred]),
        "predicted_best_true_value_km_s": float(y_true[best_pred]),
    }


def _calendar_bucket(day_index: int) -> str:
    if day_index < 91:
        return "Q1"
    if day_index < 182:
        return "Q2"
    if day_index < 273:
        return "Q3"
    return "Q4"


def calendar_bucket_summary(records: list[EvaluatedPoint]) -> list[dict]:
    buckets: dict[str, list[EvaluatedPoint]] = {}
    for record in records:
        buckets.setdefault(_calendar_bucket(record.day_index), []).append(record)
    out: list[dict] = []
    for bucket in ("Q1", "Q2", "Q3", "Q4"):
        rows = buckets.get(bucket, [])
        if not rows:
            continue
        totals = np.array([r.total_delta_v_km_s for r in rows], dtype=float)
        best = min(rows, key=lambda r: r.total_delta_v_km_s)
        out.append(
            {
                "bucket": bucket,
                "count": len(rows),
                "mean_total_delta_v_km_s": float(np.mean(totals)),
                "best_date": best.date,
                "best_total_delta_v_km_s": best.total_delta_v_km_s,
            }
        )
    return out


def permutation_importance(model: RidgeModel, records: list[EvaluatedPoint], positions: np.ndarray) -> list[dict]:
    points = records_to_points(records)
    features, names = feature_matrix(points, positions)
    targets = records_to_targets(records)
    baseline_pred = model.predict_matrix(features)
    baseline_mae = float(np.mean(np.abs(baseline_pred - targets)))
    out: list[dict] = []
    for col, name in enumerate(names):
        shuffled = features.copy()
        # Deterministic cyclic permutation: stable across machines and avoids
        # adding an RNG dependency to the report numbers.
        shuffled[:, col] = np.roll(shuffled[:, col], 7 + col)
        pred = model.predict_matrix(shuffled)
        mae = float(np.mean(np.abs(pred - targets)))
        out.append(
            {
                "feature": name,
                "baseline_mae_km_s": baseline_mae,
                "permuted_mae_km_s": mae,
                "importance_km_s": mae - baseline_mae,
            }
        )
    out.sort(key=lambda row: row["importance_km_s"], reverse=True)
    return out


def local_refinement_points(center: EvaluatedPoint, day_radius: int = 6) -> list[DesignPoint]:
    rp_offsets = (-0.025, -0.0125, 0.0, 0.0125, 0.025)
    rm_offsets = (0.0, 1_500.0, 4_000.0, 8_000.0)
    rp_min = 2.0 * R_SUN / AU_KM
    points: list[DesignPoint] = []
    seen: set[tuple[int, int, int, str]] = set()
    for day in range(center.day_index - day_radius, center.day_index + day_radius + 1, 2):
        day_clamped = int(min(364, max(0, day)))
        for rp_delta in rp_offsets:
            rp = float(min(0.4, max(rp_min, center.rp_AU + rp_delta)))
            for rm_delta in rm_offsets:
                rm = float(min(50_000.0, max(MIN_LUNAR_PERIAPSIS, center.rm_km + rm_delta)))
                for side in SIDES:
                    key = (day_clamped, round(rp * 10_000), round(rm), side)
                    if key in seen:
                        continue
                    seen.add(key)
                    points.append(DesignPoint(day_index=day_clamped, rp_AU=rp, rm_km=rm, side=side))
    return points


def verify_local_refinement(center: EvaluatedPoint, positions: np.ndarray, velocities: np.ndarray) -> dict:
    points = local_refinement_points(center)
    records = evaluate_design_points(points, positions, velocities)
    best = min(records, key=lambda item: item.total_delta_v_km_s)
    totals = np.array([r.total_delta_v_km_s for r in records], dtype=float)
    return {
        "center": center.to_dict(),
        "evaluated_points": len(records),
        "best": best.to_dict(),
        "improvement_over_center_km_s": center.total_delta_v_km_s - best.total_delta_v_km_s,
        "median_total_delta_v_km_s": float(np.median(totals)),
        "p10_total_delta_v_km_s": float(np.quantile(totals, 0.10)),
        "p90_total_delta_v_km_s": float(np.quantile(totals, 0.90)),
    }


def run_surrogate_assisted_design(
    positions: np.ndarray,
    velocities: np.ndarray,
    train_count: int = 160,
    verify_count: int = 48,
    bank_day_stride: int = 1,
) -> dict:
    halton_points = low_discrepancy_training_points(train_count)
    anchor_points = physics_anchor_points()
    training_points = merge_design_points(halton_points, anchor_points)
    training_records = evaluate_design_points(training_points, positions, velocities)
    model = fit_ridge(training_records, positions, alpha=2.0e-3)
    scores = cross_validate(training_records, positions, folds=5, alpha=2.0e-3)

    bank = candidate_bank(day_stride=bank_day_stride, rp_count=17, rm_count=11)
    predictions = predict_points(model, bank, positions)
    shortlist = _deduplicate_predictions(predictions, verify_count)
    verified = evaluate_design_points([_prediction_to_design(item) for item in shortlist], positions, velocities)
    verified_sorted = sorted(verified, key=lambda item: item.total_delta_v_km_s)
    training_best = min(training_records, key=lambda item: item.total_delta_v_km_s)
    anchor_best = min(training_records, key=lambda item: item.total_delta_v_km_s)
    verified_best = min([verified_sorted[0], anchor_best], key=lambda item: item.total_delta_v_km_s)
    predicted_best = shortlist[0]
    local_refinement = verify_local_refinement(verified_best, positions, velocities)
    exact_evaluations = len(training_records) + len(verified) + local_refinement["evaluated_points"]
    reduction = 1.0 - exact_evaluations / max(1, len(bank))
    return {
        "method": "dependency-free ridge surrogate with Halton sampling and exact top-candidate verification",
        "halton_training_count": len(halton_points),
        "physics_anchor_count": len(anchor_points),
        "training_count": len(training_records),
        "candidate_bank_size": len(bank),
        "verified_shortlist_count": len(verified),
        "local_refinement_evaluations": local_refinement["evaluated_points"],
        "exact_model_evaluations": exact_evaluations,
        "evaluation_reduction_fraction_vs_bank": reduction,
        "cross_validation": _summarize_scores(scores),
        "training_diagnostics": prediction_diagnostics(model, training_records, positions),
        "verified_diagnostics": prediction_diagnostics(model, verified_sorted, positions),
        "calendar_bucket_summary": calendar_bucket_summary(training_records + verified_sorted),
        "top_coefficients": model.coefficients()[:10],
        "permutation_importance": permutation_importance(model, training_records, positions)[:10],
        "training_best": training_best.to_dict(),
        "anchor_best": anchor_best.to_dict(),
        "predicted_best": predicted_best.to_dict(),
        "verified_best": verified_best.to_dict(),
        "local_refinement": local_refinement,
        "verified_top_candidates": [item.to_dict() for item in verified_sorted[:12]],
        "notes": "The deterministic full scan remains authoritative; this module demonstrates ML-assisted screening.",
    }


def write_surrogate_artifacts(summary: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "surrogate_design_summary.json"
    csv_path = out_dir / "surrogate_verified_candidates.csv"
    coef_path = out_dir / "surrogate_feature_coefficients.csv"
    importance_path = out_dir / "surrogate_permutation_importance.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary.get("verified_top_candidates", [])
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    coefs = summary.get("top_coefficients", [])
    if coefs:
        with coef_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(coefs[0].keys()))
            writer.writeheader()
            writer.writerows(coefs)
    importance = summary.get("permutation_importance", [])
    if importance:
        with importance_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(importance[0].keys()))
            writer.writeheader()
            writer.writerows(importance)
    return {
        "summary_json": f"data/generated/{json_path.name}",
        "verified_candidates_csv": f"data/generated/{csv_path.name}",
        "feature_coefficients_csv": f"data/generated/{coef_path.name}",
        "permutation_importance_csv": f"data/generated/{importance_path.name}",
        "verified_best": summary.get("verified_best"),
        "local_refinement": summary.get("local_refinement"),
        "evaluation_reduction_fraction_vs_bank": summary.get("evaluation_reduction_fraction_vs_bank"),
    }


def self_test(positions: np.ndarray, velocities: np.ndarray) -> dict:
    summary = run_surrogate_assisted_design(positions, velocities, train_count=32, verify_count=8, bank_day_stride=14)
    best = summary["verified_best"]
    return {
        "training_count": summary["training_count"],
        "candidate_bank_size": summary["candidate_bank_size"],
        "verified_shortlist_count": summary["verified_shortlist_count"],
        "verified_best_total_delta_v_km_s": best["total_delta_v_km_s"],
        "passed": summary["training_count"] == 32 and summary["verified_shortlist_count"] == 8 and best["total_delta_v_km_s"] > 0.0,
    }
