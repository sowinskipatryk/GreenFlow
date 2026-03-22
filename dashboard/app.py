import sys
from pathlib import Path

# Streamlit runs this file as a synthetic __main__ module (no __package__), so
# `from .components` would raise ImportError. Put the repo root on sys.path and
# import the `dashboard` package by absolute name instead.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.data_loader import (
    ExperimentData,
    list_experiments,
    load_tripinfos,
    load_stats,
    load_stopinfos,
    filter_tripinfos,
)
from dashboard.components import comparison, emissions, overview, temporal

RESULTS_DIR = Path(__file__).parent.parent / "simulation" / "results"

st.set_page_config(
    page_title="GreenFlow Dashboard",
    page_icon="\U0001f6a6",
    layout="wide",
)

# --- Sidebar ---
st.sidebar.title("GreenFlow")
st.sidebar.markdown("Analiza symulacji SUMO")

experiments = list_experiments(RESULTS_DIR)

if not experiments:
    st.error("Brak eksperymentów w folderze simulation/results/")
    st.stop()

selected = st.sidebar.multiselect(
    "Eksperymenty",
    experiments,
    default=[experiments[0]],
)

if not selected:
    st.info("Wybierz co najmniej jeden eksperyment w panelu bocznym.")
    st.stop()

# --- Global filters ---
st.sidebar.markdown("---")
st.sidebar.subheader("Filtry")

ALL_VTYPES = ["car", "car_ev", "motorcycle", "truck", "bus", "tram", "emergency"]
vehicle_filter = st.sidebar.multiselect(
    "Typy pojazdów",
    ALL_VTYPES,
    default=ALL_VTYPES,
)

finished_only = st.sidebar.checkbox("Tylko ukończone podróże", value=True)

time_filter = st.sidebar.slider(
    "Zakres czasu wyjazdu [s]",
    min_value=0,
    max_value=3600,
    value=(0, 3600),
)

# --- Preload data with filters ---
data: dict[str, ExperimentData] = {}
for exp in selected:
    exp_path = RESULTS_DIR / exp
    raw_trips = load_tripinfos(str(exp_path))
    data[exp] = ExperimentData(
        trips=filter_tripinfos(raw_trips, vehicle_filter, finished_only, time_filter),
        trips_raw=raw_trips,
        stats=load_stats(str(exp_path)),
        stopinfos=load_stopinfos(str(exp_path)),
    )

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Podsumowanie", "Porównanie", "Emisje", "Analiza czasowa",
])

with tab1:
    overview.render(data)
with tab2:
    comparison.render(data)
with tab3:
    emissions.render(data)
with tab4:
    temporal.render(data)
