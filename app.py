"""
Zwigato Delivery Delay Predictor
--------------------------------
Streamlit app for Business Analytics Assignment 2 (predictive analytics &
managerial AI adoption). Loads the linear regression (delivery time, minutes)
and logistic regression (probability of being late, >30 min) models trained
in the accompanying Colab notebook, and reproduces that notebook's exact
feature-engineering pipeline so a live order's inputs line up with the
columns each model was trained on.

Model files expected alongside this script:
    linear_regression_model.joblib
    logistic_regression_model.joblib
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Zwigato Delivery Delay Predictor", page_icon="🛵", layout="centered")

LATE_THRESHOLD = 30  # minutes — same business rule used to train the logistic model


# ----------------------------------------------------------------------------
# Load models
# ----------------------------------------------------------------------------
@st.cache_resource
def load_models():
    linear_model = joblib.load("linear_regression_model.joblib")
    logistic_model = joblib.load("logistic_regression_model.joblib")
    return linear_model, logistic_model


linear_model, logistic_model = load_models()
FEATURE_ORDER = list(linear_model.feature_names_in_)  # identical for both models


# ----------------------------------------------------------------------------
# Feature engineering — mirrors the notebook's clean_data / feature_engineering /
# pd.get_dummies(drop_first=True) pipeline exactly, for a single order.
# ----------------------------------------------------------------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius, km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 17:
        return "Afternoon"
    elif 17 <= hour < 21:
        return "Evening"
    else:
        return "Night"


def build_feature_row(order: dict) -> pd.DataFrame:
    """Turn one order's raw inputs into the one-hot-encoded row the models expect."""

    distance_km = haversine_distance(
        order["restaurant_lat"], order["restaurant_lon"],
        order["delivery_lat"], order["delivery_lon"],
    )

    order_dt = order["order_datetime"]
    day_of_week = order_dt.weekday()  # Monday=0 .. Sunday=6, matches Order_Date.dt.dayofweek
    month = order_dt.month
    hour = order_dt.hour
    time_of_day = get_time_of_day(hour)

    row = {
        "Delivery_person_Age": order["age"],
        "Delivery_person_Ratings": order["ratings"],
        "Restaurant_latitude": order["restaurant_lat"],
        "Restaurant_longitude": order["restaurant_lon"],
        "Delivery_location_latitude": order["delivery_lat"],
        "Delivery_location_longitude": order["delivery_lon"],
        "Vehicle_condition": order["vehicle_condition"],
        "multiple_deliveries": order["multiple_deliveries"],
        "Order_Preparation_Time": order["prep_time"],
        "Distance_km": distance_km,
    }

    # One-hot blocks — category names and the dropped (reference) category in each
    # match pd.get_dummies(..., drop_first=True) exactly as run in the notebook.
    # Missing-value tokens ("NaN "/"conditions NaN") became the literal category
    # "NaN" (Weatherconditions) or "None" (Festival, City, Road_traffic_density)
    # in the training data, and those are included as selectable options below
    # so the app can reproduce an order with missing information if needed.

    one_hot_blocks = {
        "Weatherconditions": (order["weather"], ["Fog", "NaN", "Sandstorms", "Stormy", "Sunny", "Windy"]),  # ref: Cloudy
        "Road_traffic_density": (order["traffic"], ["Jam", "Low", "Medium"]),  # ref: High (None never appears in this data)
        "Type_of_order": (order["order_type"], ["Drinks ", "Meal ", "Snack "]),  # ref: Buffet
        "Type_of_vehicle": (order["vehicle_type"], ["motorcycle ", "scooter "]),  # ref: bicycle / electric_scooter
        "Festival": (order["festival"], ["Yes"]),  # ref: No (None never appears in this data)
        "City": (order["city"], ["Semi-Urban", "Urban"]),  # ref: Metropolitian (None never appears in this data)
        "Time_of_Day": (time_of_day, ["Evening", "Morning", "Night"]),  # ref: Afternoon
        "Order_DayOfWeek": (day_of_week, [1, 2, 3, 4, 5, 6]),  # ref: 0 (Monday)
        "Order_Month": (month, [3, 4]),  # ref: any other month (only Mar/Apr seen in training data)
        "Order_Hour": (hour, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),  # ref: 0-7
    }

    for prefix, (value, categories) in one_hot_blocks.items():
        for cat in categories:
            col = f"{prefix}_{cat}"
            row[col] = 1 if value == cat else 0

    df_row = pd.DataFrame([row])
    # Reindex to the exact column order/set the models were trained on;
    # any column not explicitly set above (there shouldn't be any) is filled 0.
    df_row = df_row.reindex(columns=FEATURE_ORDER, fill_value=0)
    return df_row


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🛵 Zwigato Delivery Delay Predictor")
st.caption(
    "Predicts expected delivery time and the probability an order will be late "
    f"(later than {LATE_THRESHOLD} minutes), to help operations managers set "
    "delivery promises and flag orders needing attention."
)

with st.form("order_form"):
    st.subheader("Order details")

    col1, col2 = st.columns(2)
    with col1:
        order_date = st.date_input("Order date", value=dt.date.today())
        order_time = st.time_input("Order time", value=dt.time(19, 0))
    with col2:
        prep_time = st.number_input(
            "Estimated kitchen preparation time (minutes)",
            min_value=0.0, max_value=60.0, value=15.0, step=1.0,
            help="Expected time between the order being placed and the rider picking it up.",
        )
        multiple_deliveries = st.selectbox("Multiple deliveries on this trip", [0, 1, 2, 3], index=1)

    st.subheader("Locations")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Restaurant**")
        restaurant_lat = st.number_input("Restaurant latitude", value=12.9716, format="%.6f")
        restaurant_lon = st.number_input("Restaurant longitude", value=77.5946, format="%.6f")
    with col4:
        st.markdown("**Delivery location**")
        delivery_lat = st.number_input("Delivery latitude", value=13.0500, format="%.6f")
        delivery_lon = st.number_input("Delivery longitude", value=77.6500, format="%.6f")

    st.subheader("Rider")
    col5, col6, col7 = st.columns(3)
    with col5:
        age = st.number_input("Rider age", min_value=15, max_value=50, value=30)
    with col6:
        ratings = st.number_input("Rider rating", min_value=1.0, max_value=6.0, value=4.7, step=0.1)
    with col7:
        vehicle_condition = st.selectbox("Vehicle condition (0=poor, 3=best)", [0, 1, 2, 3], index=1)

    st.subheader("Conditions")
    col8, col9 = st.columns(2)
    with col8:
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "NaN"], index=0)
        traffic = st.selectbox("Road traffic density", ["Low", "Medium", "High", "Jam"], index=1)
        vehicle_type = st.selectbox("Vehicle type", ["motorcycle ", "scooter ", "electric_scooter ", "bicycle "], index=0)
    with col9:
        order_type = st.selectbox("Type of order", ["Snack ", "Meal ", "Drinks ", "Buffet "], index=0)
        festival = st.selectbox("Festival day", ["No", "Yes"], index=0)
        city = st.selectbox("City type", ["Urban", "Metropolitian", "Semi-Urban"], index=1)

    submitted = st.form_submit_button("Predict delivery outcome")

if submitted:
    order = {
        "order_datetime": dt.datetime.combine(order_date, order_time),
        "prep_time": prep_time,
        "multiple_deliveries": multiple_deliveries,
        "restaurant_lat": restaurant_lat,
        "restaurant_lon": restaurant_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
        "age": age,
        "ratings": ratings,
        "vehicle_condition": vehicle_condition,
        "weather": weather,
        "traffic": traffic,
        "vehicle_type": vehicle_type,
        "order_type": order_type,
        "festival": festival,
        "city": city,
    }

    X = build_feature_row(order)

    predicted_minutes = float(linear_model.predict(X)[0])
    late_proba = float(logistic_model.predict_proba(X)[0][1])
    is_late = late_proba >= 0.5

    st.divider()
    st.subheader("Prediction")

    res1, res2 = st.columns(2)
    with res1:
        st.metric("Predicted delivery time", f"{predicted_minutes:.0f} min")
    with res2:
        st.metric(
            f"Probability of being late (> {LATE_THRESHOLD} min)",
            f"{late_proba * 100:.0f}%",
            delta="Likely late" if is_late else "Likely on time",
            delta_color="inverse" if is_late else "normal",
        )

    if is_late:
        st.warning(
            f"This order is flagged as **likely late** — predicted delivery time "
            f"is {predicted_minutes:.0f} minutes against the {LATE_THRESHOLD}-minute service level."
        )
    else:
        st.success(
            f"This order is flagged as **likely on time** — predicted delivery time "
            f"is {predicted_minutes:.0f} minutes against the {LATE_THRESHOLD}-minute service level."
        )

    # -- Explanation: top drivers, from the logistic model's coefficients --
    st.subheader("What's driving this prediction")
    coefs = pd.Series(logistic_model.coef_[0], index=FEATURE_ORDER)
    contributions = (coefs * X.iloc[0]).sort_values(key=np.abs, ascending=False)
    top_contributions = contributions[contributions != 0].head(6)

    if len(top_contributions) > 0:
        explain_df = pd.DataFrame({
            "Factor": top_contributions.index,
            "Effect on late risk": ["Increases risk" if v > 0 else "Decreases risk" for v in top_contributions],
        })
        st.table(explain_df)
        st.caption(
            "Factors are the active inputs for this order with the largest effect, "
            "positive or negative, on the logistic model's late-delivery score."
        )
    else:
        st.caption("No single input stands out strongly for this particular order.")

    with st.expander("See the exact feature values sent to the models"):
        st.dataframe(X.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Built for the Zwigato delivery-delay case study. Linear regression estimates delivery "
    "minutes; logistic regression estimates the probability of a late delivery, defined as "
    f"delivery time exceeding {LATE_THRESHOLD} minutes — a business rule set for this study, "
    "not a fixed SLA."
)
