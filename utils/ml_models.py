import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    IsolationForest
)

from sklearn.cluster import KMeans

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.metrics import (
    r2_score,
    accuracy_score,
    classification_report
)

from sklearn.neighbors import NearestNeighbors


# =====================================================
# DATA PREPARATION
# =====================================================

def prepare_data(df):

    df = df.copy()

    categorical_cols = [
        "Industry",
        "Region"
    ]

    encoders = {}

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(
            df[col]
        )

        encoders[col] = le

    return df, encoders


# =====================================================
# VALUATION PREDICTION
# =====================================================

def train_valuation_model(df):

    df, encoders = prepare_data(df)

    features = [
        "Funding Amount (M USD)",
        "Funding Rounds",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Industry",
        "Region"
    ]

    X = df[features]

    y = df["Valuation (M USD)"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    score = r2_score(
        y_test,
        predictions
    )

    return {
        "model": model,
        "score": score,
        "features": features,
        "encoders": encoders
    }


# =====================================================
# PREDICT VALUATION
# =====================================================

def predict_valuation(
        model,
        input_data
):

    prediction = model.predict(
        [input_data]
    )[0]

    return round(
        prediction,
        2
    )


# =====================================================
# EXIT STATUS PREDICTION
# =====================================================

def train_exit_prediction(df):

    if "Exit Status" not in df.columns:
        return None

    df, encoders = prepare_data(df)

    target_encoder = LabelEncoder()

    df["Exit Status"] = (
        target_encoder.fit_transform(
            df["Exit Status"]
        )
    )

    features = [
        "Funding Amount (M USD)",
        "Funding Rounds",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Industry",
        "Region"
    ]

    X = df[features]

    y = df["Exit Status"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        preds
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "target_encoder": target_encoder,
        "features": features
    }


# =====================================================
# UNICORN PREDICTION
# =====================================================

def train_unicorn_model(df):

    df = df.copy()

    df["Unicorn"] = np.where(
        df["Valuation (M USD)"] >= 1000,
        1,
        0
    )

    df, encoders = prepare_data(df)

    features = [
        "Funding Amount (M USD)",
        "Funding Rounds",
        "Revenue (M USD)",
        "Employees",
        "Market Share (%)",
        "Industry",
        "Region"
    ]

    X = df[features]

    y = df["Unicorn"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = accuracy_score(
        y_test,
        model.predict(X_test)
    )

    return {
        "model": model,
        "accuracy": accuracy,
        "features": features
    }


# =====================================================
# STARTUP SUCCESS SCORE
# =====================================================

def calculate_success_score(df):

    df = df.copy()

    revenue_score = (
        df["Revenue (M USD)"]
        /
        df["Revenue (M USD)"].max()
    )

    funding_score = (
        df["Funding Amount (M USD)"]
        /
        df["Funding Amount (M USD)"].max()
    )

    market_score = (
        df["Market Share (%)"]
        /
        df["Market Share (%)"].max()
    )

    valuation_score = (
        df["Valuation (M USD)"]
        /
        df["Valuation (M USD)"].max()
    )

    df["Success Score"] = (

        revenue_score * 0.30
        +
        funding_score * 0.20
        +
        market_score * 0.25
        +
        valuation_score * 0.25

    ) * 100

    return df


# =====================================================
# STARTUP CLUSTERING
# =====================================================

def startup_segmentation(df):

    data = df.copy()

    features = [
        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Valuation (M USD)"
    ]

    scaler = StandardScaler()

    X = scaler.fit_transform(
        data[features]
    )

    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )

    data["Cluster"] = (
        kmeans.fit_predict(X)
    )

    return data


# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def valuation_feature_importance(df):

    result = train_valuation_model(df)

    model = result["model"]

    importance = pd.DataFrame({

        "Feature":
        result["features"],

        "Importance":
        model.feature_importances_

    })

    importance = (
        importance
        .sort_values(
            "Importance",
            ascending=False
        )
    )

    return importance


# =====================================================
# ANOMALY DETECTION
# =====================================================

def detect_anomalous_startups(df):

    features = [

        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Valuation (M USD)"

    ]

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    df = df.copy()

    df["Anomaly"] = (
        iso.fit_predict(
            df[features]
        )
    )

    return df[
        df["Anomaly"] == -1
    ]


# =====================================================
# INVESTOR RECOMMENDATION
# =====================================================

def investment_recommendations(df):

    df = calculate_success_score(df)

    recommendations = (

        df.sort_values(
            "Success Score",
            ascending=False
        )

    )

    return recommendations.head(20)


# =====================================================
# SIMILAR STARTUPS
# =====================================================

def find_similar_startups(
        df,
        startup_name,
        n_neighbors=5
):

    data = df.copy()

    features = [

        "Funding Amount (M USD)",
        "Revenue (M USD)",
        "Valuation (M USD)"

    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        data[features]
    )

    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1
    )

    model.fit(scaled)

    idx = data[
        data["Startup Name"]
        ==
        startup_name
    ].index[0]

    distances, indices = (
        model.kneighbors(
            [scaled[idx]]
        )
    )

    similar = data.iloc[
        indices[0][1:]
    ]

    return similar


# =====================================================
# FUNDING REQUIREMENT PREDICTION
# =====================================================

def estimate_required_funding(

        target_revenue,
        efficiency_ratio=2

):

    funding_needed = (

        target_revenue
        /
        efficiency_ratio

    )

    return round(
        funding_needed,
        2
    )


# =====================================================
# EXECUTIVE AI REPORT
# =====================================================

def generate_ml_report(df):

    valuation_model = (
        train_valuation_model(df)
    )

    unicorn_model = (
        train_unicorn_model(df)
    )

    top_startups = (
        investment_recommendations(df)
    )

    report = {

        "valuation_model_score":
        round(
            valuation_model["score"],
            3
        ),

        "unicorn_model_accuracy":
        round(
            unicorn_model["accuracy"],
            3
        ),

        "top_investment_opportunities":
        top_startups[
            [
                "Startup Name",
                "Success Score"
            ]
        ].head(10)

    }

    return report
