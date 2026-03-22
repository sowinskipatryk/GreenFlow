from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import ExperimentData

BIN_SIZE = 300  # 5 minutes
MG_TO_KG = 1e-6


def render(data: dict[str, ExperimentData]) -> None:
    st.subheader("Analiza czasowa")

    has_data = {name: not exp.trips.empty for name, exp in data.items()}
    if not any(has_data.values()):
        st.warning("Brak danych dla wybranych filtrów.")
        return

    for name, ok in has_data.items():
        if not ok:
            st.warning(f"⚠ {name}: brak danych dla wybranych filtrów — pominięto")

    valid = {n: d for n, d in data.items() if has_data[n]}
    bins = np.arange(0, 3600 + BIN_SIZE, BIN_SIZE)
    bin_labels = [f"{int(b)}–{int(b + BIN_SIZE)}" for b in bins[:-1]]

    # --- Block 1: departures over time ---
    st.markdown("#### Wyjazdy w czasie")
    dep_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips
        trips_binned = pd.cut(trips["depart"], bins=bins, labels=bin_labels, right=False)
        counts = trips_binned.value_counts().sort_index()
        for label, count in counts.items():
            dep_rows.append({"Czas": label, "Liczba pojazdów": count, "Eksperyment": exp_name})
    if dep_rows:
        fig = px.line(
            pd.DataFrame(dep_rows),
            x="Czas",
            y="Liczba pojazdów",
            color="Eksperyment",
            title="Liczba wyjeżdżających pojazdów per 5-minutowy interwał",
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # --- Block 2: average waiting time over time ---
    st.markdown("#### Średni czas oczekiwania w czasie")
    wait_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips.copy()
        trips["bin"] = pd.cut(trips["depart"], bins=bins, labels=bin_labels, right=False)
        avg_wait = trips.groupby("bin", observed=False)["waitingTime"].mean()
        for label, val in avg_wait.items():
            if pd.notna(val):
                wait_rows.append({"Czas": label, "Śr. oczekiwanie [s]": val, "Eksperyment": exp_name})
    if wait_rows:
        fig = px.line(
            pd.DataFrame(wait_rows),
            x="Czas",
            y="Śr. oczekiwanie [s]",
            color="Eksperyment",
            title="Średni czas oczekiwania per 5-minutowy interwał",
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # --- Block 3: average timeLoss over time ---
    st.markdown("#### Średnia strata czasu w czasie")
    tl_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips.copy()
        trips["bin"] = pd.cut(trips["depart"], bins=bins, labels=bin_labels, right=False)
        avg_tl = trips.groupby("bin", observed=False)["timeLoss"].mean()
        for label, val in avg_tl.items():
            if pd.notna(val):
                tl_rows.append({"Czas": label, "Śr. timeLoss [s]": val, "Eksperyment": exp_name})
    if tl_rows:
        fig = px.line(
            pd.DataFrame(tl_rows),
            x="Czas",
            y="Śr. timeLoss [s]",
            color="Eksperyment",
            title="Średnia strata czasu per 5-minutowy interwał",
            markers=True,
        )
        fig.update_layout(margin=dict(t=40, b=20), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # --- Block 4: cumulative CO2 over time ---
    st.markdown("#### Skumulowane emisje CO₂")
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips.copy()
        finished = trips[trips["is_finished"]].sort_values("arrival").copy()
        if finished.empty:
            continue
        finished["cum_CO2_kg"] = (finished["CO2_abs"] * MG_TO_KG).cumsum()
        fig = px.area(
            finished,
            x="arrival",
            y="cum_CO2_kg",
            title=f"{exp_name}: skumulowane CO₂ w czasie",
            labels={"arrival": "Czas zakończenia podróży [s]", "cum_CO2_kg": "CO₂ [kg]"},
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
