import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ChurnAI | Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background: #0b1120;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background: #111c30;
    border-right: 1px solid #26354d;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #60a5fa;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
}

.metric-card {
    background: linear-gradient(135deg,#172554,#1e3a5f);
    padding: 23px;
    border-radius: 15px;
    border: 1px solid #334155;
    min-height: 125px;
}

.metric-title {
    color: #94a3b8;
    font-size: 14px;
}

.metric-value {
    color: white;
    font-size: 29px;
    font-weight: bold;
}

.section-title {
    color: #60a5fa;
    font-size: 23px;
    font-weight: bold;
}

div.stButton > button {
    background: linear-gradient(90deg,#2563eb,#06b6d4);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #0891b2;
    color: white;
}

.stProgress > div > div > div > div {
    background-color: #3b82f6;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load("customer_churn_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    threshold = joblib.load("threshold.pkl")

    return model, feature_names, float(threshold)


@st.cache_data
def load_data():

    if os.path.exists("churn.csv"):
        return pd.read_csv("churn.csv")

    return pd.DataFrame()


df = load_data()

model = None
feature_names = None
threshold = 0.45

try:
    model, feature_names, threshold = load_model()
    model_status = True

except Exception as e:
    model_status = False
    model_error = str(e)


# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def metric_card(title, value, description):

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-title">{description}</div>
    </div>
    """, unsafe_allow_html=True)


def get_churn_column(data):

    return (
        data["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "1", "true"])
    )


def prepare_customer(data):

    data = data.copy()

    # Feature engineering exactly as in training

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Very Long"]
    )

    data["ChargesCategory"] = pd.cut(
        data["MonthlyCharges"],
        bins=[0, 35, 70, 120],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    # Maintain original training feature order

    data = data[feature_names]

    return data


def predict_customer(customer):

    prepared = prepare_customer(customer)

    probabilities = model.predict_proba(prepared)

    classes = list(model.classes_)
    positive_index = classes.index(1)

    churn_probability = probabilities[0][positive_index]

    prediction = int(churn_probability >= threshold)

    return prediction, churn_probability


def risk_level(probability):

    if probability < 0.30:
        return "LOW", "🟢"

    elif probability < 0.60:
        return "MEDIUM", "🟡"

    return "HIGH", "🔴"


def retention_advice(customer, probability):

    advice = []

    if probability >= threshold:
        advice.append("Offer a personalized retention discount.")

    if customer["Contract"] == "Month-to-month":
        advice.append("Encourage switching to a yearly contract.")

    if customer["tenure"] <= 12:
        advice.append("Provide special onboarding support.")

    if customer["MonthlyCharges"] > 80:
        advice.append("Review pricing and offer a suitable plan.")

    if customer["PaymentMethod"] == "Electronic check":
        advice.append("Offer automatic payment options.")

    if not advice:
        advice.append("Continue loyalty rewards and regular engagement.")

    return advice


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.markdown("# ⚡ ChurnAI")

st.sidebar.caption("CUSTOMER INTELLIGENCE PLATFORM")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "🤖 Churn Prediction",
        "📈 Analytics",
        "📋 Prediction History",
        "🧠 Model Information",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

if model_status:
    st.sidebar.success("● ML Model Connected")
else:
    st.sidebar.error("● ML Model Not Connected")

if not df.empty:
    st.sidebar.info(f"Dataset: {len(df):,} customers")


# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">Customer Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered customer churn monitoring system</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if not df.empty and "Churn" in df.columns:

        total = len(df)

        churned = int(get_churn_column(df).sum())

        retained = total - churned

        churn_rate = churned / total * 100

    else:

        total = churned = retained = 0
        churn_rate = 0

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Customers",
            f"{total:,}",
            "Total customers analyzed"
        )

    with c2:
        metric_card(
            "Churned Customers",
            f"{churned:,}",
            "Customers who left"
        )

    with c3:
        metric_card(
            "Retained Customers",
            f"{retained:,}",
            "Customers retained"
        )

    with c4:
        metric_card(
            "Churn Rate",
            f"{churn_rate:.2f}%",
            "Overall churn percentage"
        )

    st.write("")

    st.markdown(
        '<div class="section-title">Business Overview</div>',
        unsafe_allow_html=True
    )

    if not df.empty:

        col1, col2 = st.columns(2)

        with col1:

            fig = px.pie(
                names=["Churned", "Retained"],
                values=[churned, retained],
                hole=0.65,
                title="Customer Retention Distribution",
                color_discrete_sequence=["#ef4444", "#06b6d4"]
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        with col2:

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=churn_rate,
                    title={"text": "Churn Rate"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#3b82f6"},
                        "steps": [
                            {"range": [0, 30], "color": "#164e63"},
                            {"range": [30, 60], "color": "#713f12"},
                            {"range": [60, 100], "color": "#7f1d1d"}
                        ]
                    }
                )
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Recent Customer Records")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

    else:

        st.warning("Dataset not found.")


# ==========================================
# CHURN PREDICTION
# ==========================================

elif page == "🤖 Churn Prediction":

    st.markdown(
        '<div class="main-title">AI Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Analyze customer behavior using Machine Learning</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if not model_status:

        st.error(
            "Model files could not be loaded. "
            "Check your pickle files and project folder."
        )

        st.stop()

    st.markdown("### 👤 Customer Information")

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["Yes", "No", "No phone service"]
            )

        with col2:

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

            device = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly Charges ($)",
                min_value=0.0,
                max_value=150.0,
                value=70.0,
                step=1.0
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=float(monthly * tenure),
                step=10.0
            )

        st.write("")

        submitted = st.form_submit_button(
            "🚀 Predict Customer Churn",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{

            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total_charges

        }])

        try:

            prediction, probability = predict_customer(customer)

            risk, icon = risk_level(probability)

            st.markdown("---")

            st.markdown(
                '<div class="section-title">Prediction Results</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )

            with c2:

                st.metric(
                    "Risk Level",
                    f"{icon} {risk}"
                )

            with c3:

                st.metric(
                    "Prediction",
                    "Churn" if prediction == 1 else "Retained"
                )

            st.progress(float(probability))

            if prediction == 1:

                st.error(
                    "⚠️ This customer is predicted to churn."
                )

            else:

                st.success(
                    "✅ This customer is predicted to stay."
                )

            st.markdown("### 💡 Retention Recommendations")

            recommendations = retention_advice(
                customer.iloc[0],
                probability
            )

            for recommendation in recommendations:

                st.info("💡 " + recommendation)

            # Save prediction history

            record = customer.iloc[0].to_dict()

            record["Churn Probability"] = round(
                probability * 100, 2
            )

            record["Prediction"] = (
                "Churn" if prediction == 1 else "Retained"
            )

            record["Risk Level"] = risk

            st.session_state.history.append(record)

        except Exception as e:

            st.error(f"Prediction Error: {str(e)}")


# ==========================================
# ANALYTICS
# ==========================================

elif page == "📈 Analytics":

    st.markdown(
        '<div class="main-title">Customer Analytics</div>',
        unsafe_allow_html=True
    )

    if df.empty:

        st.warning("Dataset not found.")

    else:

        churn_data = df.copy()

        churn_data["Churn Status"] = np.where(
            get_churn_column(churn_data),
            "Churned",
            "Retained"
        )

        st.subheader("Churn Distribution")

        fig = px.histogram(
            churn_data,
            x="Churn Status",
            color="Churn Status",
            color_discrete_map={
                "Churned": "#ef4444",
                "Retained": "#06b6d4"
            }
        )

        fig.update_layout(
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            if "Contract" in df.columns:

                fig = px.histogram(
                    churn_data,
                    x="Contract",
                    color="Churn Status",
                    barmode="group",
                    title="Churn by Contract Type"
                )

                fig.update_layout(
                    paper_bgcolor="#0b1120",
                    font_color="white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        with col2:

            if "InternetService" in df.columns:

                fig = px.histogram(
                    churn_data,
                    x="InternetService",
                    color="Churn Status",
                    barmode="group",
                    title="Churn by Internet Service"
                )

                fig.update_layout(
                    paper_bgcolor="#0b1120",
                    font_color="white"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        if "tenure" in df.columns:

            fig = px.histogram(
                churn_data,
                x="tenure",
                color="Churn Status",
                barmode="overlay",
                title="Churn by Customer Tenure"
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if "MonthlyCharges" in df.columns:

            fig = px.box(
                churn_data,
                x="Churn Status",
                y="MonthlyCharges",
                color="Churn Status",
                title="Monthly Charges vs Churn"
            )

            fig.update_layout(
                paper_bgcolor="#0b1120",
                font_color="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ==========================================
# PREDICTION HISTORY
# ==========================================

elif page == "📋 Prediction History":

    st.markdown(
        '<div class="main-title">Prediction History</div>',
        unsafe_allow_html=True
    )

    history = st.session_state.history

    if history:

        history_df = pd.DataFrame(history)

        st.metric(
            "Total Predictions",
            len(history_df)
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Prediction Report",
            data=csv,
            file_name="churn_prediction_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info(
            "No predictions yet. Go to Churn Prediction to get started."
        )


# ==========================================
# MODEL INFORMATION
# ==========================================

elif page == "🧠 Model Information":

    st.markdown(
        '<div class="main-title">Machine Learning Model</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Random Forest Classifier")

    st.write(
        "Our project uses a tuned Random Forest classification pipeline "
        "to identify customers likely to discontinue their services."
    )

    if model_status:

        st.success("Trained model successfully loaded.")

        st.metric(
            "Prediction Threshold",
            f"{threshold:.2f}"
        )

        st.markdown("### Model Performance")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Accuracy", "75.80%", "Test dataset")

        with c2:
            metric_card("Precision", "53.04%", "Test dataset")

        with c3:
            metric_card("Recall", "77.01%", "Test dataset")

        with c4:
            metric_card("F1 Score", "62.81%", "Test dataset")

        st.info(
            "These are the previously reported evaluation metrics "
            "from your model training."
        )

    else:

        st.error("Model files could not be loaded.")

    st.markdown("---")

    st.subheader("ML Workflow")

    steps = [
        "Data Collection",
        "Data Cleaning",
        "Exploratory Data Analysis",
        "Feature Engineering",
        "Data Preprocessing",
        "Random Forest Training",
        "Hyperparameter Tuning",
        "Model Evaluation",
        "Churn Prediction"
    ]

    for i, step in enumerate(steps, 1):

        st.write(f"**{i}.** {step}")


# ==========================================
# ABOUT
# ==========================================

elif page == "ℹ️ About":

    st.markdown(
        '<div class="main-title">About ChurnAI</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.markdown("""
    ## ⚡ ChurnAI

    An intelligent customer churn prediction platform
    powered by Machine Learning.

    ### Project Objectives

    - Predict customers likely to leave a service.
    - Analyze customer behavior.
    - Identify important churn patterns.
    - Help businesses improve customer retention.
    - Provide data-driven business insights.

    ### Technology Stack

    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Random Forest
    - Streamlit
    - Plotly
    - Joblib

    ### Machine Learning

    Tuned Random Forest Classifier with
    preprocessing pipeline and optimized prediction threshold.

    """)

    st.success(
        "Turning customer data into actionable business insights."
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    """
    <center style="color:#64748b;">
        ChurnAI | Customer Intelligence Platform
        <br>
        Built with Python, Machine Learning & Streamlit
    </center>
    """,
    unsafe_allow_html=True
)
