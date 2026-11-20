import os
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="P13 Road-Accident Hotspot Miner",
    page_icon="🚦",
    layout="wide",
)

# ---------------------------------------------------------------------
# Paths & Cache-Safe Asset Loaders
# ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HOTSPOTS_PATH = os.path.join(BASE_DIR, "Data", "hotspots_ca_fl_dbscan.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "Models", "lgbm_severity_bundle.joblib")


@st.cache_data
def _read_hotspots(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_resource
def _read_model_bundle(path: str):
    return joblib.load(path)


def load_hotspots():
    if not os.path.exists(HOTSPOTS_PATH):
        return None
    return _read_hotspots(HOTSPOTS_PATH)


def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return _read_model_bundle(MODEL_PATH)


# ---------------------------------------------------------------------
# Helpers for Derived Features & Model Input Alignment
# ---------------------------------------------------------------------
PRESETS = {
    "Los Angeles — I-10 / I-110 (CA)": ("CA", 34.0392, -118.2693),
    "Miami — Golden Glades I-95 (FL)": ("FL", 25.9095, -80.2092),
    "San Francisco Bay Area — I-80 (CA)": ("CA", 37.8339, -122.3017),
    "Orlando — I-4 Corridor (FL)": ("FL", 28.4498, -81.4737),
    "Custom Coordinates": None,
}

DAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

SEVERITY_LABELS = {
    1: "Severity 1 — Minimal Delay",
    2: "Severity 2 — Moderate Delay (Typical Surface Street)",
    3: "Severity 3 — Significant Impact (Typical Freeway/Interstate)",
    4: "Severity 4 — Severe / Extended Closure",
}


def derive_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def derive_time_of_day(hour: int) -> str:
    if 6 <= hour <= 9:
        return "Morning Rush (6-9)"
    if 10 <= hour <= 15:
        return "Midday (10-15)"
    if 16 <= hour <= 19:
        return "Evening Rush (16-19)"
    if 20 <= hour <= 23:
        return "Late_Evening (20-23)"
    return "Overnight (0-5)"


def derive_visibility_bin(vis: float) -> str:
    if vis < 2.0:
        return "Low (<2 mi)"
    if vis <= 7.0:
        return "Moderate (2-7 mi)"
    return "High (>7 mi)"


def build_model_input(pipeline, raw_values: dict) -> pd.DataFrame:
    """Align input DataFrame columns and dtypes to the exact fitted pipeline."""
    expected_cols = list(getattr(pipeline, "feature_names_in_", raw_values.keys()))
    row = {col: raw_values.get(col, 0) for col in expected_cols}

    # Ensure DayOfWeek matches whether it was fitted as int (0-6) or string
    if "DayOfWeek" in row:
        num_cols = set()
        for _, step in getattr(pipeline, "steps", []):
            if hasattr(step, "transformers_"):
                for name, _, cols in step.transformers_:
                    if "num" in name.lower() and isinstance(cols, (list, tuple, np.ndarray)):
                        num_cols.update(cols)
        if "DayOfWeek" in num_cols and isinstance(row["DayOfWeek"], str):
            row["DayOfWeek"] = DAY_NAMES.index(row["DayOfWeek"])

    return pd.DataFrame([row], columns=expected_cols)


# ---------------------------------------------------------------------
# Header & Tabs
# ---------------------------------------------------------------------
st.title("🚦 Road-Accident Hotspot Miner (DS-311 Project 13)")
st.caption(
    "DBSCAN Spatial Blackspots (CA & FL) · Pre-Crash Severity Predictor (LightGBM) · "
    "Temporally Validated Apriori Risk Rules"
)

tab1, tab2, tab3 = st.tabs([
    "📍 Hotspot Map (DBSCAN)",
    "⚡ Live Severity Predictor",
    "📋 High-Risk Rules (Apriori)",
])


# =====================================================================
# TAB 1: DBSCAN HOTSPOT MAP
# =====================================================================
with tab1:
    df_hotspots = load_hotspots()
    if df_hotspots is None:
        st.error(
            f"Missing `{HOTSPOTS_PATH}`. Make sure `hotspots_ca_fl_dbscan.parquet` "
            "is inside the `Data/` folder."
        )
    else:
        col_f1, col_f2, col_f3 = st.columns([1.2, 1.5, 1.5])
        with col_f1:
            state_opts = ["All (CA & FL)"] + sorted(df_hotspots["State"].dropna().unique().tolist())
            selected_state = st.selectbox("Filter State", state_opts)
        with col_f2:
            min_crashes = st.slider(
                "Minimum Crashes per Hotspot",
                min_value=250,
                max_value=int(df_hotspots["total_crashes"].quantile(0.95)),
                value=500,
                step=50,
            )
        with col_f3:
            min_high_sev = st.slider(
                "Minimum High-Severity Share (%)",
                min_value=0.0,
                max_value=60.0,
                value=0.0,
                step=5.0,
            )

        filtered = df_hotspots[
            (df_hotspots["total_crashes"] >= min_crashes)
            & (df_hotspots["high_sev_pct"] >= min_high_sev)
        ].copy()
        if selected_state != "All (CA & FL)":
            filtered = filtered[filtered["State"] == selected_state]

        m1, m2, m3 = st.columns(3)
        m1.metric("Matching Hotspots", f"{len(filtered):,}")
        m2.metric("Total Crashes Covered", f"{int(filtered['total_crashes'].sum()):,}")
        m3.metric(
            "Avg High-Severity Share",
            f"{filtered['high_sev_pct'].mean():.1f}%" if len(filtered) else "0.0%",
        )

        if len(filtered) == 0:
            st.warning("No hotspots match those filter thresholds. Lower the sliders to see clusters.")
        else:
            filtered["radius_m"] = np.clip(np.sqrt(filtered["total_crashes"]) * 28, 350, 3500)
            filtered["color_r"] = np.clip((filtered["high_sev_pct"] / 50.0) * 255, 80, 255).astype(int)
            filtered["color_g"] = np.clip((1.0 - filtered["high_sev_pct"] / 60.0) * 190, 30, 190).astype(int)

            view_lat = float(filtered["centroid_lat"].mean())
            view_lng = float(filtered["centroid_lng"].mean())
            zoom = 5.3 if selected_state != "All (CA & FL)" else 3.6

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered,
                get_position="[centroid_lng, centroid_lat]",
                get_radius="radius_m",
                get_fill_color="[color_r, color_g, 40, 185]",
                pickable=True,
                auto_highlight=True,
            )
            tooltip = {
                "html": (
                    "<b>{State} Hotspot #{db_label}</b><br/>"
                    "Total Crashes: {total_crashes}<br/>"
                    "High-Sev (3-4): {high_sev_pct}%<br/>"
                    "Coords: ({centroid_lat}, {centroid_lng})"
                ),
                "style": {"backgroundColor": "#1e1e1e", "color": "white"},
            }
            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=pdk.ViewState(
                        latitude=view_lat, longitude=view_lng, zoom=zoom
                    ),
                    tooltip=tooltip,
                ),
                use_container_width=True,
            )

            st.subheader("Top 15 Hotspots by Crash Volume")
            display_cols = [
                c
                for c in [
                    "State",
                    "db_label",
                    "centroid_lat",
                    "centroid_lng",
                    "total_crashes",
                    "high_sev_pct",
                    "pre22_crashes",
                    "post22_crashes",
                ]
                if c in filtered.columns
            ]
            st.dataframe(
                filtered.sort_values("total_crashes", ascending=False)[display_cols].head(15),
                use_container_width=True,
                hide_index=True,
            )


# =====================================================================
# TAB 2: LIVE PRE-CRASH SEVERITY PREDICTOR
# =====================================================================
with tab2:
    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            f"Missing `{MODEL_PATH}`. Make sure `lgbm_severity_bundle.joblib` "
            "is inside the `Models/` folder."
        )
    else:
        pipeline = bundle["pipeline"]
        multipliers = np.asarray(bundle["multipliers"], dtype=float)

        st.markdown(
            "Predicts **traffic-impact severity (1–4)** using only pre-crash context "
            "(location, time, weather, and road class — zero post-crash `Distance(mi)` leakage)."
        )

        col_loc, col_wx, col_road = st.columns([1.15, 1.15, 0.95])

        # --- Column 1: Location & Time ---
        with col_loc:
            st.subheader("Location & Time")
            preset_name = st.selectbox("Location Preset", list(PRESETS.keys()))
            if PRESETS[preset_name] is not None:
                state, lat, lng = PRESETS[preset_name]
                st.caption(f"Locked to preset → **State:** `{state}` · **Lat:** `{lat:.4f}` · **Lng:** `{lng:.4f}`")
            else:
                state = st.selectbox("State", ["CA", "FL", "TX", "NY", "NC", "SC", "PA", "VA", "OR", "MN"])
                lat = st.number_input("Latitude", value=34.0392, min_value=24.0, max_value=50.0, format="%.4f")
                lng = st.number_input("Longitude", value=-118.2693, min_value=-125.0, max_value=-66.0, format="%.4f")

            hour = st.slider("Hour of Day (0–23)", 0, 23, 17)
            month = st.slider("Month (1–12)", 1, 12, 7)
            day_name = st.selectbox("Day of Week", DAY_NAMES, index=0)

            day_idx = DAY_NAMES.index(day_name)
            is_weekend = int(day_idx >= 5)
            sunrise_sunset = "Day" if 6 <= hour < 19 else "Night"
            season = derive_season(month)
            time_of_day = derive_time_of_day(hour)
            st.caption(
                f"Auto-derived → **{sunrise_sunset}** · **{time_of_day}** · **{season}**"
            )

        # --- Column 2: Weather ---
        with col_wx:
            st.subheader("Weather Conditions")
            weather_cat = st.selectbox(
                "Weather Category",
                ["Clear", "Cloudy", "Rain", "Fog/Haze", "Snow/Ice"],
            )

            if weather_cat in ("Rain", "Snow/Ice"):
                precip = st.slider(
                    "Precipitation (in/hr)",
                    min_value=0.01,
                    max_value=1.50,
                    value=0.15,
                    step=0.01,
                )
            else:
                precip = 0.0
                st.caption("Precipitation auto-set to **0.00 in/hr** for dry/fog conditions.")

            default_vis = {"Clear": 10.0, "Cloudy": 10.0, "Rain": 5.0, "Fog/Haze": 1.5, "Snow/Ice": 2.0}[weather_cat]
            visibility = st.slider("Visibility (mi)", 0.2, 10.0, default_vis, step=0.2)
            temp_f = st.slider("Temperature (°F)", 0.0, 115.0, 72.0, step=1.0)
            humidity = st.slider("Humidity (%)", 5, 100, 60)
            wind_mph = st.slider("Wind Speed (mph)", 0.0, 45.0, 7.0, step=1.0)
            pressure = st.slider("Barometric Pressure (inHg)", 28.5, 31.0, 29.95, step=0.05)

        # --- Column 3: Roadway Features ---
        with col_road:
            st.subheader("Roadway Context")
            st.caption(
                "Leave unchecked for an **open Interstate/freeway** (higher Severity 3 impact). "
                "Check **Traffic Signal** or **Crossing** for urban surface streets (typically Severity 2)."
            )
            junction = int(st.checkbox("Junction / Interchange", value=False))
            traffic_signal = int(st.checkbox("Traffic Signal", value=False))
            crossing = int(st.checkbox("Crossing (Crosswalk)", value=False))
            stop_sign = int(st.checkbox("Stop Sign", value=False))
            station = int(st.checkbox("Transit Station", value=False))
            railway = int(st.checkbox("Railway Crossing", value=False))
            roundabout = int(st.checkbox("Roundabout", value=False))
            give_way = int(st.checkbox("Give Way / Yield", value=False))

        # --- Build Feature Row & Predict ---
        raw_features = {
            "Start_Lat": float(lat),
            "Start_Lng": float(lng),
            "Temperature(F)": float(temp_f),
            "Wind_Chill(F)": float(temp_f),
            "Humidity(%)": float(humidity),
            "Pressure(in)": float(pressure),
            "Visibility(mi)": float(visibility),
            "Wind_Speed(mph)": float(wind_mph),
            "Precipitation(in)": float(precip),
            "Hour": int(hour),
            "Month": int(month),
            "DayOfWeek": int(day_idx),
            "Is_Weekend": int(is_weekend),
            "State": state,
            "Season": season,
            "Time_of_Day": time_of_day,
            "Weather_Category": weather_cat,
            "Visibility_Bin": derive_visibility_bin(visibility),
            "Sunrise_Sunset": sunrise_sunset,
            "Amenity": 0,
            "Bump": 0,
            "Crossing": crossing,
            "Give_Way": give_way,
            "Junction": junction,
            "No_Exit": 0,
            "Railway": railway,
            "Roundabout": roundabout,
            "Station": station,
            "Stop": stop_sign,
            "Traffic_Calming": 0,
            "Traffic_Signal": traffic_signal,
            "Turning_Loop": 0,
        }

        X_input = build_model_input(pipeline, raw_features)
        probs = pipeline.predict_proba(X_input)[0]
        classes = list(pipeline.classes_)
        pred_idx = int(np.argmax(probs * multipliers))
        pred_class = int(classes[pred_idx])

        high_sev_prob = sum(
            p for c, p in zip(classes, probs) if int(c) in (3, 4)
        ) * 100.0

        st.markdown("### Prediction Result")
        res_l, res_r = st.columns([1.1, 1.9])
        with res_l:
            if pred_class >= 3:
                st.warning(f"**Predicted Class:** {SEVERITY_LABELS[pred_class]}")
            else:
                st.success(f"**Predicted Class:** {SEVERITY_LABELS[pred_class]}")
            st.metric("Combined High-Severity (3–4) Probability", f"{high_sev_prob:.1f}%")
            st.caption(
                "Probabilities come directly from the 31-feature LightGBM pipeline "
                "(Stratified Macro-F1: 0.4662); class label uses validation-tuned thresholds."
            )
        with res_r:
            prob_df = pd.DataFrame({
                "Severity": [f"Severity {int(c)}" for c in classes],
                "Probability (%)": np.round(probs * 100.0, 2),
            }).set_index("Severity")
            st.bar_chart(prob_df, use_container_width=True)


# =====================================================================
# TAB 3: TEMPORALLY VALIDATED APRIORI RULES
# =====================================================================
with tab3:
    st.subheader("Top High-Severity Association Rules (`Severity 3–4`)")
    st.caption(
        "Mined strictly on the Stratified Training split (n = 397,440; base rate 19.55%) "
        "and cross-checked on the held-out Test split and Temporal 2022–23 split."
    )

    rules_df = pd.DataFrame([
        {
            "Antecedent Conditions": "Junction + Summer",
            "Train Lift": 1.655,
            "Test Lift": 1.572,
            "Pre-2022 Lift": 1.580,
            "2022–23 Lift": 1.618,
            "Temporal Status": "Stable ✅",
        },
        {
            "Antecedent Conditions": "Summer + Rain",
            "Train Lift": 1.454,
            "Test Lift": 1.342,
            "Pre-2022 Lift": 1.328,
            "2022–23 Lift": 1.928,
            "Temporal Status": "Stable (Strengthened) ✅",
        },
        {
            "Antecedent Conditions": "Junction + Late Evening (20–23h)",
            "Train Lift": 1.414,
            "Test Lift": 1.328,
            "Pre-2022 Lift": 1.358,
            "2022–23 Lift": 1.309,
            "Temporal Status": "Stable ✅",
        },
        {
            "Antecedent Conditions": "Summer + Moderate Visibility (2–7 mi)",
            "Train Lift": 1.412,
            "Test Lift": 1.274,
            "Pre-2022 Lift": 1.240,
            "2022–23 Lift": 2.100,
            "Temporal Status": "Stable (Strengthened) ✅",
        },
        {
            "Antecedent Conditions": "Weekend + Junction",
            "Train Lift": 1.405,
            "Test Lift": 1.385,
            "Pre-2022 Lift": 1.475,
            "2022–23 Lift": 0.992,
            "Temporal Status": "Drifted (Source Shift) ⚠️",
        },
    ])
    st.dataframe(rules_df, use_container_width=True, hide_index=True)
