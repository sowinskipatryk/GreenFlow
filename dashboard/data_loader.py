import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st


@dataclass
class ExperimentData:
    """Data container passed from app.py to component render functions."""
    trips: pd.DataFrame
    trips_raw: pd.DataFrame
    stats: dict
    stopinfos: pd.DataFrame


TRIPINFO_NUMERIC_COLS = [
    "depart", "arrival", "duration", "routeLength",
    "waitingTime", "waitingCount", "stopTime", "timeLoss",
    "departDelay", "speedFactor",
    "CO_abs", "CO2_abs", "HC_abs", "PMx_abs", "NOx_abs",
    "fuel_abs", "electricity_abs",
]


def list_experiments(results_dir: Path) -> list[str]:
    if not results_dir.is_dir():
        return []
    return sorted(
        d.name
        for d in results_dir.iterdir()
        if d.is_dir() and (d / "tripinfos.xml").exists()
    )


@st.cache_data(show_spinner="Ładowanie tripinfos...")
def load_tripinfos(experiment_path: str) -> pd.DataFrame:
    path = Path(experiment_path) / "tripinfos.xml"
    if not path.exists():
        return _empty_tripinfos()

    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[dict] = []
    for elem in root.iter("tripinfo"):
        row = dict(elem.attrib)
        emissions_elem = elem.find("emissions")
        if emissions_elem is not None:
            row.update(emissions_elem.attrib)
        rows.append(row)

    if not rows:
        return _empty_tripinfos()

    df = pd.DataFrame(rows)
    for col in TRIPINFO_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["is_finished"] = df["arrival"] >= 0
    df["avg_speed"] = df["routeLength"] / df["duration"].replace(0, float("nan"))

    return df


def _empty_tripinfos() -> pd.DataFrame:
    cols = ["id", "vType"] + TRIPINFO_NUMERIC_COLS + ["is_finished", "avg_speed"]
    return pd.DataFrame(columns=cols)


@st.cache_data(show_spinner="Ładowanie stats...")
def load_stats(experiment_path: str) -> dict:
    path = Path(experiment_path) / "stats.xml"
    if not path.exists():
        return _empty_stats()

    tree = ET.parse(path)
    root = tree.getroot()

    performance = root.find("performance")
    vehicles = root.find("vehicles")
    teleports = root.find("teleports")
    safety = root.find("safety")
    trip_stats = root.find("vehicleTripStatistics")

    if vehicles is None:
        return _empty_stats()

    sim_duration = float(performance.get("duration", "3600")) if performance is not None else 3600.0
    inserted = int(vehicles.get("inserted", "0"))
    running = int(vehicles.get("running", "0"))

    result = {
        "loaded": int(vehicles.get("loaded", "0")),
        "inserted": inserted,
        "running": running,
        "waiting": int(vehicles.get("waiting", "0")),

        "teleports_total": int(teleports.get("total", "0")) if teleports is not None else 0,
        "teleports_jam": int(teleports.get("jam", "0")) if teleports is not None else 0,
        "teleports_yield": int(teleports.get("yield", "0")) if teleports is not None else 0,
        "teleports_wrongLane": int(teleports.get("wrongLane", "0")) if teleports is not None else 0,

        "collisions": int(safety.get("collisions", "0")) if safety is not None else 0,
        "emergencyStops": int(safety.get("emergencyStops", "0")) if safety is not None else 0,
        "emergencyBraking": int(safety.get("emergencyBraking", "0")) if safety is not None else 0,

        "simulation_duration": sim_duration,
    }

    if trip_stats is not None:
        result.update({
            "trip_count": int(trip_stats.get("count", "0")),
            "avg_routeLength": float(trip_stats.get("routeLength", "0")),
            "avg_speed": float(trip_stats.get("speed", "0")),
            "avg_duration": float(trip_stats.get("duration", "0")),
            "avg_waitingTime": float(trip_stats.get("waitingTime", "0")),
            "avg_timeLoss": float(trip_stats.get("timeLoss", "0")),
            "avg_departDelay": float(trip_stats.get("departDelay", "0")),
            "departDelayWaiting": float(trip_stats.get("departDelayWaiting", "0")),
            "totalTravelTime": float(trip_stats.get("totalTravelTime", "0")),
            "totalDepartDelay": float(trip_stats.get("totalDepartDelay", "0")),
        })
    else:
        result.update({
            "trip_count": 0, "avg_routeLength": 0.0, "avg_speed": 0.0,
            "avg_duration": 0.0, "avg_waitingTime": 0.0, "avg_timeLoss": 0.0,
            "avg_departDelay": 0.0, "departDelayWaiting": 0.0,
            "totalTravelTime": 0.0, "totalDepartDelay": 0.0,
        })

    minutes = sim_duration / 60.0
    result["throughput"] = inserted / minutes if minutes > 0 else 0.0
    result["completion_rate"] = ((inserted - running) / inserted * 100) if inserted > 0 else 0.0

    return result


def _empty_stats() -> dict:
    return {
        "loaded": 0, "inserted": 0, "running": 0, "waiting": 0,
        "teleports_total": 0, "teleports_jam": 0, "teleports_yield": 0, "teleports_wrongLane": 0,
        "collisions": 0, "emergencyStops": 0, "emergencyBraking": 0,
        "trip_count": 0, "avg_routeLength": 0.0, "avg_speed": 0.0,
        "avg_duration": 0.0, "avg_waitingTime": 0.0, "avg_timeLoss": 0.0,
        "avg_departDelay": 0.0, "departDelayWaiting": 0.0,
        "totalTravelTime": 0.0, "totalDepartDelay": 0.0,
        "simulation_duration": 3600.0, "throughput": 0.0, "completion_rate": 0.0,
    }


@st.cache_data(show_spinner="Ładowanie stopinfos...")
def load_stopinfos(experiment_path: str) -> pd.DataFrame:
    """Stub — baseline stopinfos.xml is empty. Returns empty DataFrame."""
    path = Path(experiment_path) / "stopinfos.xml"
    if not path.exists():
        return pd.DataFrame()

    tree = ET.parse(path)
    root = tree.getroot()

    rows: list[dict] = []
    for elem in root.iter("stopinfo"):
        rows.append(dict(elem.attrib))

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def filter_tripinfos(
    df: pd.DataFrame,
    vehicle_types: list[str] | None = None,
    finished_only: bool = True,
    time_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = pd.Series(True, index=df.index)

    if vehicle_types is not None:
        mask &= df["vType"].isin(vehicle_types)

    if finished_only:
        mask &= df["is_finished"]

    if time_range is not None:
        mask &= df["depart"].between(time_range[0], time_range[1])

    return df.loc[mask].reset_index(drop=True)


def aggregate_trip_metrics(df: pd.DataFrame, sim_duration: float) -> dict:
    if df.empty:
        return {
            "avg_waitingTime": 0.0, "avg_speed": 0.0, "avg_timeLoss": 0.0,
            "avg_departDelay": 0.0, "trip_count": 0, "throughput": 0.0,
        }
    minutes = sim_duration / 60.0
    return {
        "avg_waitingTime": float(df["waitingTime"].mean()),
        "avg_speed": float(df["avg_speed"].mean()),
        "avg_timeLoss": float(df["timeLoss"].mean()),
        "avg_departDelay": float(df["departDelay"].mean()),
        "trip_count": len(df),
        "throughput": len(df) / minutes if minutes > 0 else 0.0,
    }
