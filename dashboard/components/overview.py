from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..data_loader import ExperimentData


def render(data: dict[str, ExperimentData]) -> None:
    for exp_name, exp_data in data.items():
        st.subheader(f"📊 {exp_name}")
        stats = exp_data.stats

        # Row 1 — primary KPI
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Pojazdy", f"{stats['inserted']} / {stats['loaded']}")
        c2.metric("Ukończenie", f"{stats['completion_rate']:.1f}%")
        c3.metric("Śr. oczekiwanie", f"{stats['avg_waitingTime']:.1f} s")
        c4.metric("Śr. prędkość", f"{stats['avg_speed']:.2f} m/s")
        c5.metric("Teleporty", stats["teleports_total"])
        c6.metric("Kolizje", stats["collisions"])

        # Row 2 — context KPI
        trips = exp_data.trips
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Śr. opóźnienie wyjazdu", f"{stats['avg_departDelay']:.1f} s")
        c2.metric("Pojazdy oczekujące", stats["waiting"])
        c3.metric("Przepustowość", f"{stats['throughput']:.1f} poj/min")
        if not trips.empty:
            tl_ratio = (trips["timeLoss"] / trips["duration"].replace(0, float("nan"))).mean()
            c4.metric("Strata czasu", f"{tl_ratio * 100:.1f}%" if pd.notna(tl_ratio) else "—", help="timeLoss / duration")
        else:
            c4.metric("Strata czasu", "—", help="timeLoss / duration")

        # Row 3 — charts (skip if filtered trips are empty)
        if trips.empty:
            st.warning(f"⚠ {exp_name}: brak danych dla wybranych filtrów")
        else:
            col_left, col_right = st.columns(2)
            with col_left:
                fig = px.histogram(
                    trips,
                    x="waitingTime",
                    nbins=40,
                    title="Rozkład czasu oczekiwania",
                    labels={"waitingTime": "Czas oczekiwania [s]", "count": "Liczba pojazdów"},
                )
                fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                vtype_counts = trips["vType"].value_counts().reset_index()
                vtype_counts.columns = ["vType", "count"]
                fig = px.pie(
                    vtype_counts,
                    values="count",
                    names="vType",
                    title="Rozkład typów pojazdów",
                )
                fig.update_layout(margin=dict(t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Row 4 — timeLoss histogram
        if not trips.empty:
            fig = px.histogram(
                trips,
                x="timeLoss",
                nbins=40,
                title="Rozkład straty czasu",
                labels={"timeLoss": "Strata czasu [s]", "count": "Liczba pojazdów"},
            )
            fig.update_layout(showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Row 5 — teleport details
        with st.expander("Szczegóły teleportów"):
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Zatory (jam)", stats["teleports_jam"])
            tc2.metric("Ustępowanie (yield)", stats["teleports_yield"])
            tc3.metric("Zły pas (wrongLane)", stats["teleports_wrongLane"])

        st.markdown("---")
