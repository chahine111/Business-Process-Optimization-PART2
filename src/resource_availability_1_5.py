"""
src/resource_availability_1_5.py

Learns resource working schedules from historical logs.
Basic: 2-week even/odd cycle with 1h buckets.
Advanced: month + weekday with 2h buckets + daily presence filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd


def _load_and_repair_minimal(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        csv_path,
        usecols=["Action", "org:resource", "time:timestamp"],
        low_memory=False,
    )

    # BPI 2017 has some rows where columns collapse into "Action"
    broken = df["org:resource"].isna() & df["Action"].astype(str).str.contains(",", regex=False)
    if broken.any():
        parts = df.loc[broken, "Action"].astype(str).str.split(",", n=18, expand=True)
        # empirical: index 1 = resource, index 6 = timestamp
        df.loc[broken, "org:resource"] = parts[1].values
        df.loc[broken, "time:timestamp"] = parts[6].values
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["time:timestamp", "org:resource"]).copy()
    df["org:resource"] = df["org:resource"].astype(str).str.strip()
    return df


@dataclass(frozen=True)
class AvailabilityArtifactBasic:
    tau: float
    avail_index: Dict[Tuple[int, int, int], List[str]]  # (week_type, weekday, hour) -> resources


@dataclass(frozen=True)
class AvailabilityArtifactAdvanced:
    tau_month: float
    present_by_date: Dict[object, Set[str]]             # date -> resources present that day
    monthly_index: Dict[Tuple[int, int, int], List[str]]  # (month, weekday, bucket2h) -> resources


def _train_basic(df_min: pd.DataFrame, tau: float) -> AvailabilityArtifactBasic:
    ts = df_min["time:timestamp"]
    iso = ts.dt.isocalendar()

    tmp = pd.DataFrame(
        {
            "org:resource": df_min["org:resource"].astype("string"),
            "iso_year": iso["year"].astype(np.int32),
            "iso_week": iso["week"].astype(np.int16),
            "weekday": ts.dt.weekday.astype(np.int8),
            "hour": ts.dt.hour.astype(np.int8),
        }
    )

    tmp["week_id"] = (tmp["iso_year"] * 100 + tmp["iso_week"]).astype(np.int32)
    tmp["week_type"] = (tmp["iso_week"] % 2).astype(np.int8)

    # total unique weeks of each type
    week_counts = (
        tmp[["week_id", "week_type"]]
        .drop_duplicates()
        .groupby("week_type", observed=True)["week_id"]
        .nunique()
        .sort_index()
    )

    presence = tmp[["org:resource", "week_type", "week_id", "weekday", "hour"]].drop_duplicates()
    worked = (
        presence.groupby(["org:resource", "week_type", "weekday", "hour"], observed=True)["week_id"]
        .nunique()
        .reset_index(name="worked_weeks")
    )

    worked["total_weeks"] = worked["week_type"].map(week_counts).astype(np.int16)
    worked["p_worked"] = worked["worked_weeks"] / worked["total_weeks"]
    worked["available"] = worked["p_worked"] >= float(tau)

    # build lookup index
    avail_index: Dict[Tuple[int, int, int], List[str]] = {}
    for (wt, wd, hr), sub in worked[worked["available"]].groupby(
        ["week_type", "weekday", "hour"], observed=True
    ):
        avail_index[(int(wt), int(wd), int(hr))] = sub["org:resource"].astype(str).tolist()

    return AvailabilityArtifactBasic(tau=float(tau), avail_index=avail_index)


def _train_advanced(df_min: pd.DataFrame, tau_month: float) -> AvailabilityArtifactAdvanced:
    ts = df_min["time:timestamp"]

    tmp = pd.DataFrame(
        {
            "org:resource": df_min["org:resource"].astype("string"),
            "date": ts.dt.date,
            "month": ts.dt.month.astype(np.int8),
            "weekday": ts.dt.weekday.astype(np.int8),
            "bucket2h": (ts.dt.hour // 2).astype(np.int8),
        }
    )

    # 1. which resources had at least one event on each date
    present_days = tmp.drop_duplicates(subset=["org:resource", "date", "month", "weekday"])
    present_by_date = (
        present_days.groupby("date", observed=True)["org:resource"]
        .apply(lambda s: set(s.astype(str).tolist()))
        .to_dict()
    )

    # 2. denominator: total present days per resource/month/weekday
    denom_present = (
        present_days.groupby(["org:resource", "month", "weekday"], observed=True)["date"]
        .nunique()
        .rename("present_days")
        .reset_index()
    )

    # 3. numerator: days where they appeared in this 2h bucket
    presence_day_bucket = tmp.drop_duplicates(
        subset=["org:resource", "date", "month", "weekday", "bucket2h"]
    )
    num_bucket = (
        presence_day_bucket.groupby(["org:resource", "month", "weekday", "bucket2h"], observed=True)["date"]
        .nunique()
        .rename("days_in_bucket")
        .reset_index()
    )

    # 4. probability = days in bucket / total present days
    monthly_model = num_bucket.merge(
        denom_present,
        on=["org:resource", "month", "weekday"],
        how="left",
    )
    monthly_model["p_worked"] = monthly_model["days_in_bucket"] / monthly_model["present_days"]
    monthly_model["available"] = monthly_model["p_worked"] >= float(tau_month)

    # 5. build lookup index
    monthly_index = (
        monthly_model[monthly_model["available"]]
        .groupby(["month", "weekday", "bucket2h"], observed=True)["org:resource"]
        .apply(lambda s: s.astype(str).tolist())
        .to_dict()
    )

    return AvailabilityArtifactAdvanced(
        tau_month=float(tau_month),
        present_by_date=present_by_date,
        monthly_index=monthly_index,
    )


class ResourceAvailabilityModel:

    def __init__(
        self,
        csv_path: str,
        mode: str = "basic",
        *,
        tau: float = 0.50,
        tau_month: float = 0.50,
        cache_path: Optional[str] = None,
        year_filter: int = 2016,
    ):
        self.csv_path = str(csv_path)
        self.mode = mode.lower().strip()
        if self.mode not in {"basic", "advanced"}:
            raise ValueError("Mode must be 'basic' or 'advanced'")

        self.tau = float(tau)
        self.tau_month = float(tau_month)
        self.year_filter = int(year_filter)

        # default cache path
        if cache_path is None:
            project_root = Path(__file__).resolve().parents[1]
            models_dir = project_root / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            cache_path = str(models_dir / f"resource_availability_model_1_5_{self.mode}.pkl")

        self.cache_path = cache_path
        self.artifact = self._load_or_train()

    def _load_or_train(self):
        p = Path(self.cache_path)
        if p.exists():
            try:
                return joblib.load(p)
            except Exception:
                pass

        print(f"[ResourceAvailability] Training new model (Mode: {self.mode})...")
        df = _load_and_repair_minimal(self.csv_path)

        # only use 2016 data for training (stable period)
        start = pd.Timestamp(f"{self.year_filter}-01-01", tz="UTC")
        end = pd.Timestamp(f"{self.year_filter + 1}-01-01", tz="UTC")
        df = df[(df["time:timestamp"] >= start) & (df["time:timestamp"] < end)].copy()

        if self.mode == "basic":
            art = _train_basic(df, tau=self.tau)
        else:
            art = _train_advanced(df, tau_month=self.tau_month)

        try:
            joblib.dump(art, p)
            print(f"[ResourceAvailability] Model saved to {p}")
        except Exception:
            pass
        return art

    def get_available_resources(self, t) -> List[str]:
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        if self.mode == "basic":
            week_type = int(ts.isocalendar().week) % 2
            weekday = int(ts.weekday())
            hour = int(ts.hour)
            return list(self.artifact.avail_index.get((week_type, weekday, hour), []))

        day = ts.date()
        # must have been present on this calendar day
        present_set = self.artifact.present_by_date.get(day, set())
        if not present_set:
            return []

        month = int(ts.month)
        weekday = int(ts.weekday())
        bucket2h = int(ts.hour // 2)

        candidates = self.artifact.monthly_index.get((month, weekday, bucket2h), [])
        if not candidates:
            return []

        # intersection: present today AND active in this 2h bucket
        return [r for r in candidates if r in present_set]