from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import ExperimentData

MG_TO_KG = 1e-6
MG_TO_G = 1e-3


def render(data: dict[str, ExperimentData]) -> None:
    st.subheader("Analiza emisji")

    has_data = {name: not exp.trips.empty for name, exp in data.items()}
    if not any(has_data.values()):
        st.warning("Brak danych dla wybranych filtrów.")
        return

    for name, ok in has_data.items():
        if not ok:
            st.warning(f"⚠ {name}: brak danych dla wybranych filtrów — pominięto")

    valid = {n: d for n, d in data.items() if has_data[n]}

    # --- Block 1: summary KPI ---
    st.markdown("#### Sumaryczne emisje")
    cols = st.columns(len(valid))
    for col, (exp_name, exp_data) in zip(cols, valid.items()):
        trips = exp_data.trips
        with col:
            st.markdown(f"**{exp_name}**")
            co2_kg = trips["CO2_abs"].sum() * MG_TO_KG
            co_g = trips["CO_abs"].sum() * MG_TO_G
            nox_g = trips["NOx_abs"].sum() * MG_TO_G
            pmx_g = trips["PMx_abs"].sum() * MG_TO_G
            st.metric("CO₂", f"{co2_kg:,.1f} kg")
            st.metric("CO", f"{co_g:,.1f} g")
            st.metric("NOₓ", f"{nox_g:,.1f} g")
            st.metric("PMₓ", f"{pmx_g:,.1f} g")

    # --- Block 2: comparison bar chart ---
    st.markdown("#### Porównanie emisji między eksperymentami")
    summary_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips
        summary_rows.append({
            "Eksperyment": exp_name,
            "CO₂ [kg]": trips["CO2_abs"].sum() * MG_TO_KG,
            "CO [g]": trips["CO_abs"].sum() * MG_TO_G,
            "NOₓ [g]": trips["NOx_abs"].sum() * MG_TO_G,
            "PMₓ [g]": trips["PMx_abs"].sum() * MG_TO_G,
        })
    summary_df = pd.DataFrame(summary_rows)
    melted = summary_df.melt(id_vars="Eksperyment", var_name="Emisja", value_name="Wartość")
    fig = px.bar(
        melted,
        x="Emisja",
        y="Wartość",
        color="Eksperyment",
        barmode="group",
        title="Łączne emisje per eksperyment",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Block 3: emissions per vehicle type (stacked bar) ---
    st.markdown("#### Emisje CO₂ per typ pojazdu")
    vtype_rows = []
    for exp_name, exp_data in valid.items():
        by_vtype = exp_data.trips.groupby("vType")["CO2_abs"].sum() * MG_TO_KG
        for vtype, val in by_vtype.items():
            vtype_rows.append({"Eksperyment": exp_name, "Typ pojazdu": vtype, "CO₂ [kg]": val})
    if vtype_rows:
        fig = px.bar(
            pd.DataFrame(vtype_rows),
            x="Eksperyment",
            y="CO₂ [kg]",
            color="Typ pojazdu",
            barmode="stack",
            title="CO₂ per typ pojazdu",
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- Block 4: fuel vs electricity ---
    st.markdown("#### Zużycie energii: paliwo vs elektryczność")
    energy_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips
        energy_rows.append({
            "Eksperyment": exp_name,
            "Paliwo [g]": trips["fuel_abs"].sum() * MG_TO_G,
            "Elektryczność [Wh]": trips["electricity_abs"].sum(),
        })
    energy_df = pd.DataFrame(energy_rows)
    melted_e = energy_df.melt(id_vars="Eksperyment", var_name="Typ energii", value_name="Wartość")
    fig = px.bar(
        melted_e,
        x="Eksperyment",
        y="Wartość",
        color="Typ energii",
        barmode="group",
        title="Paliwo vs Elektryczność",
    )
    fig.update_layout(margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- Block 5: CO2 intensity per vehicle type ---
    st.markdown("#### Intensywność emisji CO₂ per typ pojazdu")
    intensity_rows = []
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips
        valid_trips = trips[trips["routeLength"] > 0]
        if valid_trips.empty:
            continue
        co2_g_per_km = (valid_trips["CO2_abs"] * MG_TO_G) / (valid_trips["routeLength"] / 1000)
        by_vtype = co2_g_per_km.groupby(valid_trips["vType"]).mean()
        for vtype, val in by_vtype.items():
            intensity_rows.append({"Eksperyment": exp_name, "Typ pojazdu": vtype, "CO₂ [g/km]": val})
    if intensity_rows:
        fig = px.bar(
            pd.DataFrame(intensity_rows),
            x="Typ pojazdu",
            y="CO₂ [g/km]",
            color="Eksperyment",
            barmode="group",
            title="Średnia emisja CO₂ na kilometr per typ pojazdu",
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # --- Block 6: emissions per vehicle scatter ---
    st.markdown("#### Efektywność emisyjna: trasa vs CO₂")
    for exp_name, exp_data in valid.items():
        trips = exp_data.trips.copy()
        trips["CO₂ [g]"] = trips["CO2_abs"] * MG_TO_G
        fig = px.scatter(
            trips,
            x="routeLength",
            y="CO₂ [g]",
            color="vType",
            title=f"{exp_name}: długość trasy vs emisja CO₂",
            labels={"routeLength": "Długość trasy [m]", "CO₂ [g]": "CO₂ [g]"},
            opacity=0.6,
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
