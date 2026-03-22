from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import ExperimentData

MG_TO_KG = 1e-6


def _pct_change(baseline_val: float, other_val: float) -> str:
    if baseline_val == 0:
        return "—"
    pct = (other_val - baseline_val) / baseline_val * 100
    return f"{pct:+.1f}%"


def _build_comparison_row(name: str, exp_data: ExperimentData) -> dict:
    s = exp_data.stats
    co2_kg = exp_data.trips["CO2_abs"].sum() * MG_TO_KG if not exp_data.trips.empty else 0.0
    return {
        "Śr. oczekiwanie [s]": s["avg_waitingTime"],
        "Śr. prędkość [m/s]": s["avg_speed"],
        "Śr. timeLoss [s]": s["avg_timeLoss"],
        "Śr. departDelay [s]": s["avg_departDelay"],
        "Przepustowość [poj/min]": s["throughput"],
        "Ukończenie [%]": s["completion_rate"],
        "Teleporty": s["teleports_total"],
        "CO₂ łącznie [kg]": co2_kg,
    }


def render(data: dict[str, ExperimentData]) -> None:
    st.subheader("Porównanie eksperymentów")

    if len(data) < 2:
        st.info("Wybierz co najmniej 2 eksperymenty w panelu bocznym, aby zobaczyć porównanie.")
        return

    exp_names = list(data.keys())

    # Block 1 — baseline selector
    baseline_name = st.selectbox(
        "Eksperyment bazowy (baseline)",
        exp_names,
        index=0,
    )

    # Block 2 — comparison table
    metrics_per_exp = {name: _build_comparison_row(name, exp_data) for name, exp_data in data.items()}
    metric_names = list(metrics_per_exp[baseline_name].keys())
    baseline_vals = metrics_per_exp[baseline_name]

    table_data: dict[str, list] = {"Metryka": metric_names}
    for exp_name in exp_names:
        table_data[exp_name] = [metrics_per_exp[exp_name][m] for m in metric_names]

    # Add % change columns for non-baseline experiments
    for exp_name in exp_names:
        if exp_name == baseline_name:
            continue
        col_name = f"Δ {exp_name} vs {baseline_name}"
        table_data[col_name] = [
            _pct_change(baseline_vals[m], metrics_per_exp[exp_name][m])
            for m in metric_names
        ]

    comparison_df = pd.DataFrame(table_data)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    # Block 3 — charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### Kluczowe metryki")
        chart_metrics = ["Śr. oczekiwanie [s]", "Śr. prędkość [m/s]", "Przepustowość [poj/min]"]
        bar_rows = []
        for exp_name, exp_metrics in metrics_per_exp.items():
            for m in chart_metrics:
                bar_rows.append({"Eksperyment": exp_name, "Metryka": m, "Wartość": exp_metrics[m]})
        fig = px.bar(
            pd.DataFrame(bar_rows),
            x="Metryka",
            y="Wartość",
            color="Eksperyment",
            barmode="group",
        )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("##### Rozkład czasu oczekiwania")
        box_frames = []
        for exp_name, exp_data in data.items():
            if exp_data.trips.empty:
                continue
            subset = exp_data.trips[["waitingTime"]].copy()
            subset["Eksperyment"] = exp_name
            box_frames.append(subset)
        if box_frames:
            box_df = pd.concat(box_frames, ignore_index=True)
            fig = px.box(
                box_df,
                x="Eksperyment",
                y="waitingTime",
                color="Eksperyment",
                labels={"waitingTime": "Czas oczekiwania [s]"},
            )
            fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Brak danych tripinfos dla wybranych filtrów.")

    # Block 4 — per vehicle type comparison
    st.markdown("##### Śr. czas oczekiwania per typ pojazdu")
    vtype_rows = []
    for exp_name, exp_data in data.items():
        if exp_data.trips.empty:
            continue
        by_vtype = exp_data.trips.groupby("vType")["waitingTime"].mean()
        for vtype, val in by_vtype.items():
            vtype_rows.append({"Eksperyment": exp_name, "Typ pojazdu": vtype, "Śr. oczekiwanie [s]": val})
    if vtype_rows:
        fig = px.bar(
            pd.DataFrame(vtype_rows),
            x="Typ pojazdu",
            y="Śr. oczekiwanie [s]",
            color="Eksperyment",
            barmode="group",
            title="Czas oczekiwania per typ pojazdu per eksperyment",
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Brak danych tripinfos dla wybranych filtrów.")

    # Block 5 — timeLoss per vehicle type
    st.markdown("##### Śr. strata czasu (timeLoss) per typ pojazdu")
    tl_vtype_rows = []
    for exp_name, exp_data in data.items():
        if exp_data.trips.empty:
            continue
        by_vtype = exp_data.trips.groupby("vType")["timeLoss"].mean()
        for vtype, val in by_vtype.items():
            tl_vtype_rows.append({"Eksperyment": exp_name, "Typ pojazdu": vtype, "Śr. timeLoss [s]": val})
    if tl_vtype_rows:
        fig = px.bar(
            pd.DataFrame(tl_vtype_rows),
            x="Typ pojazdu",
            y="Śr. timeLoss [s]",
            color="Eksperyment",
            barmode="group",
            title="Strata czasu per typ pojazdu per eksperyment",
        )
        fig.update_layout(margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Brak danych tripinfos dla wybranych filtrów.")

    # Block 6 — trip efficiency scatter
    with st.expander("Efektywność podróży: czas vs dystans", expanded=False):
        for exp_name, exp_data in data.items():
            if exp_data.trips.empty:
                continue
            fig = px.scatter(
                exp_data.trips,
                x="routeLength",
                y="duration",
                color="vType",
                title=f"{exp_name}: długość trasy vs czas podróży",
                labels={"routeLength": "Długość trasy [m]", "duration": "Czas podróży [s]"},
                opacity=0.5,
            )
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Block 7 — waitingCount per vehicle type
    with st.expander("Śr. liczba zatrzymań per typ pojazdu", expanded=False):
        wc_vtype_rows = []
        for exp_name, exp_data in data.items():
            if exp_data.trips.empty:
                continue
            by_vtype = exp_data.trips.groupby("vType")["waitingCount"].mean()
            for vtype, val in by_vtype.items():
                wc_vtype_rows.append({"Eksperyment": exp_name, "Typ pojazdu": vtype, "Śr. liczba zatrzymań": val})
        if wc_vtype_rows:
            fig = px.bar(
                pd.DataFrame(wc_vtype_rows),
                x="Typ pojazdu",
                y="Śr. liczba zatrzymań",
                color="Eksperyment",
                barmode="group",
                title="Liczba zatrzymań per typ pojazdu per eksperyment",
            )
            fig.update_layout(margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Brak danych tripinfos dla wybranych filtrów.")
