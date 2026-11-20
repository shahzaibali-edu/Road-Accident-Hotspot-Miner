import os
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="P13 Road-Accident Hotspot Miner",
    page_icon="🚦",
    layout="wide"
)

# ---------------------------------------------------------
# Paths & Cached Asset Loaders
# ---------------------------------------------------------
HOTSPOTS_PATH = os.path.join("data", "hotspots_ca_fl_dbscan.parquet")
MODEL_PATH = os.path.join("models", "lgbm_severity_bundle.joblib")

@st.cache_data
def load_hotspots():
    if not os.path.exists(HOTSPOTS_PATH):
        return None
    return pd.read_parquet(HOTSPOTS_PATH)

@st.cache_resource
def load_model_bundle():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

hotspots_df = load_hotspots()
model_bundle = load_model_bundle()

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🚦 Road-Accident Hotspot Miner (DS-311 Project 13)")
st.caption(
    "DBSCAN Spatial Blackspots (CA & FL) · Pre-Crash Severity Predictor (LightGBM) · "
    "Temporally Validated Apriori Risk Rules"
)

tab1, tab2, tab3 = st.tabs([
    "📍 Hotspot Map (DBSCAN)",
    "⚡ Live Severity Predictor",
    "📋 High-Risk Rules (Apriori)"
])

# ---------------------------------------------------------
# TAB 1: Interactive DBSCAN Hotspot Explorer
# ---------------------------------------------------------
with tab1:
    if hotspots_df is None:
        st.error(f"Missing `{HOTSPOTS_PATH}`. Copy it from Google Drive into the `data/` folder.")
    else:
        c1, c2, c3 = st.columns([1, 1, 1])
        state_choice = c1.selectbox("Select State", ["CA", "FL"], index=0)
        min_crashes = c2.slider("Minimum Crashes per Hotspot", 250, 5000, 500, step=50)
        min_sev_pct = c3.slider("Minimum High-Severity Share (%)", 0, 60, 0, step=5)

        filt = hotspots_df[
            (hotspots_df["State"] == state_choice) &
            (hotspots_df["total_crashes"] >= min_crashes) &
            (hotspots_df["high_sev_pct"] >= min_sev_pct)
        ].copy()

        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Hotspots Shown", f"{len(filt):,}")
        k2.metric("Total Crashes Covered", f"{int(filt['total_crashes'].sum()):,}")
        k3.metric("Avg High-Severity %", f"{filt['high_sev_pct'].mean():.1f}%" if len(filt) else "0.0%")
        k4.metric("DBSCAN Mean Radius", "0.50 km (CA)" if state_choice == "CA" else "0.36 km (FL)")

        if len(filt) > 0:
            # Color scale: yellow -> red by high_sev_pct
            sev_norm = np.clip(filt["high_sev_pct"] / 50.0, 0.0, 1.0)
            filt["r"] = 235
            filt["g"] = (220 * (1.0 - sev_norm)).astype(int)
            filt["b"] = 40
            filt["radius_m"] = np.clip(filt["total_crashes"] * 0.45, 350, 4500)
            filt["sev_str"] = filt["high_sev_pct"].round(1).astype(str) + "%"

            center_lat = 36.2 if state_choice == "CA" else 27.8
            center_lng = -119.5 if state_choice == "CA" else -81.7

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filt,
                get_position="[centroid_lng, centroid_lat]",
                get_fill_color="[r, g, b, 190]",
                get_radius="radius_m",
                pickable=True,
                auto_highlight=True
            )
            view_state = pdk.ViewState(
                latitude=center_lat, longitude=center_lng, zoom=5.5, pitch=0
            )
            tooltip = {
                "html": "<b>Hotspot #{db_label} ({State})</b><br/>"
                        "Total Crashes: {total_crashes}<br/>"
                        "High-Severity (3–4): {sev_str}<br/>"
                        "Pre-2022: {pre22_crashes} | 2022–23: {post22_crashes}",
                "style": {"backgroundColor": "#1e293b", "color": "white"}
            }
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

            st.subheader(f"Top 15 Hotspots in {state_choice}")
            display_cols = [
                "db_label", "centroid_lat", "centroid_lng",
                "total_crashes", "high_sev_pct", "pre22_crashes", "post22_crashes"
            ]
            top_tbl = filt.head(15)[display_cols].copy()
            top_tbl["centroid_lat"] = top_tbl["centroid_lat"].round(4)
            top_tbl["centroid_lng"] = top_tbl["centroid_lng"].round(4)
            top_tbl["high_sev_pct"] = top_tbl["high_sev_pct"].round(1)
            st.dataframe(top_tbl, use_container_width=True, hide_index=True)
        else:
            st.warning("No hotspots match those filter thresholds. Try lowering the sliders.")

# ---------------------------------------------------------
# TAB 2: Live Crash Severity Predictor (LightGBM)
# ---------------------------------------------------------
with tab2:
    if model_bundle is None:
        st.error(f"Missing `{MODEL_PATH}`. Copy it from Google Drive into the `models/` folder.")
    else:
        st.markdown("Enter **pre-crash** conditions (no post-crash `Distance(mi)`) to estimate Severity 1–4 risk:")

        # Quick city presets for fast demos
        presets = {
            "Los Angeles (I-10/I-110)": ("CA", 34.0392, -118.2693, "US/Pacific"),
            "Oakland Bay Bridge (I-80)": ("CA", 37.8339, -122.3017, "US/Pacific"),
            "Miami (Golden Glades I-95)": ("FL", 25.9095, -80.2092, "US/Eastern"),
            "Orlando (I-4 Corridor)": ("FL", 28.4498, -81.4737, "US/Eastern"),
            "Custom Coordinates": ("CA", 34.05, -118.25, "US/Pacific")
        }
        preset_name = st.selectbox("Quick Location Preset", list(presets.keys()))
        p_state, p_lat, p_lng, p_tz = presets[preset_name]

        with st.form("predict_form"):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Location & Time**")
                state_in = st.selectbox("State", ["CA", "FL", "TX", "NY", "NC", "SC", "PA", "VA"],
                                        index=0 if p_state == "CA" else 1)
                lat_in = st.number_input("Latitude", value=float(p_lat), format="%.4f")
                lng_in = st.number_input("Longitude", value=float(p_lng), format="%.4f")
                hour_in = st.slider("Hour of Day (0–23)", 0, 23, 17)
                month_in = st.slider("Month (1–12)", 1, 12, 7)
                dow_in = st.selectbox(
                    "Day of Week",
                    [(0, "Monday"), (1, "Tuesday"), (2, "Wednesday"), (3, "Thursday"),
                     (4, "Friday"), (5, "Saturday"), (6, "Sunday")],
                    format_func=lambda x: x[1]
                )[0]

            with col_b:
                st.markdown("**Weather & Lighting**")
                weather_in = st.selectbox(
                    "Weather Category",
                    ["Clear", "Cloudy", "Rain", "Fog/Haze", "Snow/Ice", "Storm/Thunder", "Other"]
                )
                sun_in = st.selectbox("Lighting (Sunrise/Sunset)", ["Day", "Night"])
                temp_in = st.number_input("Temperature (°F)", -30.0, 130.0, 72.0)
                hum_in = st.slider("Humidity (%)", 0, 100, 60)
                pres_in = st.number_input("Pressure (inHg)", 25.0, 32.0, 29.9)
                vis_in = st.number_input("Visibility (mi)", 0.0, 20.0, 10.0)
                wind_in = st.number_input("Wind Speed (mph)", 0.0, 100.0, 8.0)
                precip_in = st.number_input("Precipitation (in)", 0.0, 5.0, 0.0, step=0.05)

            with col_c:
                st.markdown("**Roadway Features**")
                junction_in = st.checkbox("Junction / Interchange", value=True)
                signal_in = st.checkbox("Traffic Signal", value=False)
                crossing_in = st.checkbox("Crossing", value=False)
                stop_in = st.checkbox("Stop Sign", value=False)
                station_in = st.checkbox("Transit Station", value=False)
                railway_in = st.checkbox("Railway", value=False)
                roundabout_in = st.checkbox("Roundabout", value=False)
                give_way_in = st.checkbox("Give Way", value=False)

            submitted = st.form_submit_button("Predict Crash Severity", type="primary")

        if submitted:
            season_map = {
                12: "Winter", 1: "Winter", 2: "Winter",
                3: "Spring", 4: "Spring", 5: "Spring",
                6: "Summer", 7: "Summer", 8: "Summer",
                9: "Fall", 10: "Fall", 11: "Fall"
            }
            if hour_in < 6:
                tod_in = "Night (0-5)"
            elif hour_in < 10:
                tod_in = "Morning_Rush (6-9)"
            elif hour_in < 16:
                tod_in = "Midday (10-15)"
            elif hour_in < 20:
                tod_in = "Evening_Rush (16-19)"
            else:
                tod_in = "Late_Evening (20-23)"

            tz_map = {"CA": "US/Pacific", "FL": "US/Eastern", "TX": "US/Central",
                      "NY": "US/Eastern", "NC": "US/Eastern", "SC": "US/Eastern",
                      "PA": "US/Eastern", "VA": "US/Eastern"}

            input_row = pd.DataFrame([{
                "Temperature(F)": temp_in,
                "Humidity(%)": float(hum_in),
                "Pressure(in)": pres_in,
                "Visibility(mi)": vis_in,
                "Wind_Speed(mph)": wind_in,
                "Precipitation(in)": precip_in,
                "Start_Lat": lat_in,
                "Start_Lng": lng_in,
                "Hour": hour_in,
                "Month": month_in,
                "DayOfWeek": dow_in,
                "State": state_in,
                "Weather_Category": weather_in,
                "Season": season_map[month_in],
                "Time_of_Day": tod_in,
                "Sunrise_Sunset": sun_in,
                "Wind_Direction": "Unknown",
                "Timezone": tz_map.get(state_in, p_tz),
                "Amenity": False,
                "Bump": False,
                "Crossing": crossing_in,
                "Give_Way": give_way_in,
                "Junction": junction_in,
                "No_Exit": False,
                "Railway": railway_in,
                "Roundabout": roundabout_in,
                "Station": station_in,
                "Stop": stop_in,
                "Traffic_Calming": False,
                "Traffic_Signal": signal_in,
                "Is_Weekend": int(dow_in in (5, 6))
            }])

            pipe = model_bundle["pipeline"]
            mult = model_bundle["multipliers"]
            classes = pipe.named_steps["clf"].classes_

            probs = pipe.predict_proba(input_row)[0]
            pred_cls = int(classes[np.argmax(probs * mult)])
            high_sev_prob = float(probs[classes >= 3].sum() * 100.0)

            sev_labels = {
                1: "Severity 1 — Minor Delay",
                2: "Severity 2 — Moderate Delay",
                3: "Severity 3 — Significant Impact",
                4: "Severity 4 — Severe / Major Closure"
            }

            r1, r2 = st.columns([1, 1.5])
            with r1:
                st.success(f"**Predicted Class:** {sev_labels[pred_cls]}")
                st.metric("Combined High-Severity (3–4) Probability", f"{high_sev_prob:.1f}%")
            with r2:
                prob_df = pd.DataFrame({
                    "Severity": [f"Severity {c}" for c in classes],
                    "Probability (%)": np.round(probs * 100.0, 2)
                }).set_index("Severity")
                st.bar_chart(prob_df)

# ---------------------------------------------------------
# TAB 3: Temporally Validated Apriori Rules
# ---------------------------------------------------------
with tab3:
    st.subheader("Top Temporally Stable High-Severity Risk Rules (Apriori)")
    st.markdown(
        "Mined on the Stratified Training split (`n = 397,440`) with `*_Unknown` excluded, "
        "validated on the 20% Test split, and checked across reporting eras (`<=2021` vs. `2022–23`)."
    )

    rules_data = pd.DataFrame([
        {"Condition (Antecedents)": "Junction + Summer", "Train Lift": 1.66, "Test Lift": 1.57, "Lift <=2021": 1.58, "Lift 2022–23": 1.62, "Status": "Stable"},
        {"Condition (Antecedents)": "Summer + Rain", "Train Lift": 1.45, "Test Lift": 1.34, "Lift <=2021": 1.33, "Lift 2022–23": 1.93, "Status": "Stable (Stronger in 22–23)"},
        {"Condition (Antecedents)": "Junction + Late Evening (20–23)", "Train Lift": 1.41, "Test Lift": 1.33, "Lift <=2021": 1.36, "Lift 2022–23": 1.31, "Status": "Stable"},
        {"Condition (Antecedents)": "Summer + Moderate Visibility (2–7 mi)", "Train Lift": 1.41, "Test Lift": 1.27, "Lift <=2021": 1.24, "Lift 2022–23": 2.10, "Status": "Stable (Stronger in 22–23)"},
        {"Condition (Antecedents)": "Junction + Hot (80°F+)", "Train Lift": 1.53, "Test Lift": 1.40, "Lift <=2021": 1.48, "Lift 2022–23": 1.27, "Status": "Stable"},
        {"Condition (Antecedents)": "Is_Weekend + Junction", "Train Lift": 1.40, "Test Lift": 1.38, "Lift <=2021": 1.47, "Lift 2022–23": 0.99, "Status": "Drifted (Reporting Shift)"}
    ])
    st.dataframe(rules_data, use_container_width=True, hide_index=True)
    st.info(
        "Note on Reporting Shift: Between 2016–2021 and 2022–2023, US_Accidents shifted heavily toward Source1, "
        "reducing the overall Severity 3–4 base rate from 24.1% to 6.4%. Rules marked 'Stable' maintain Lift > 1.25 in both eras."
    )