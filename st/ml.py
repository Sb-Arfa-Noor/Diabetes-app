# ================= IMPORTS =================
import streamlit as st
import hashlib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
from database import create_table, get_patients, add_patient, delete_patient

# ================= LOAD ML MODELS =================
model1 = joblib.load("st/ml_models/step1.pkl")
model2 = joblib.load("st/ml_models/step2.pkl")
model3 = joblib.load("st/ml_models/step3.pkl")
# ================= FEATURE ALIGN HELPER =================
def align_features(data, model):
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)
    return df

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DiabetesCare AI System",
    layout="wide",
    initial_sidebar_state="expanded"
)
from database import create_table, add_patient, get_patients, delete_patient

create_table()



# ================= SESSION =================
if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* ================= MAIN ================= */

.stApp{
    background:#F5F7FA;
    font-family:Arial;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#062E57,#0D3B66);
    width:255px !important;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* ================= LOGO ================= */

.logo-box{
    text-align:center;
    padding:25px 10px;
}

.logo-title{
    font-size:28px;
    font-weight:bold;
}

.logo-sub{
    color:#CBD5E1;
    font-size:13px;
}

/* ================= MENU ================= */

div[data-testid="stRadio"] label{
    padding:12px;
    border-radius:12px;
    margin-bottom:5px;
    transition:0.3s;
}

div[data-testid="stRadio"] label:hover{
    background:#1565C0;
}

/* ================= TITLES ================= */

.main-title{
    font-size:34px;
    font-weight:bold;
    color:#263238;
}

.sub-title{
    color:#64748B;
    margin-top:-5px;
    margin-bottom:25px;
}

/* ================= DATE CARD ================= */

.date-card{
    background:white;
    padding:15px;
    border-radius:16px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

/* ================= METRIC CARD ================= */

.metric-card{
    background:white;
    padding:22px;
    border-radius:20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
    transition:0.3s;
}

.metric-card:hover{
    transform:translateY(-3px);
}

.metric-card div{
    line-height:1.4;
}

.metric-title{
    color:#64748B;
    font-size:14px;
    margin-bottom:10px;
}

.metric-value{
    font-size:34px;
    font-weight:700;
    margin-top:5px;
}

.metric-growth{
    color:#16A34A;
    font-size:13px;
    margin-top:6px;
}

/* ================= PANEL ================= */

.panel{
    background:white;
    padding:22px;
    border-radius:20px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

/* ================= TABLE ================= */

div[data-testid="stDataFrame"]{
    background:white;
    padding:10px;
    border-radius:18px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

/* ================= BUTTON ================= */

.stButton>button{
    background:#0D9488;
    color:white;
    border:none;
    border-radius:12px;
    padding:10px 18px;
    font-weight:bold;
    transition:0.3s;
}

.stButton>button:hover{
    background:#14B8A6;
    transform:translateY(-2px);
}

/* ================= INPUTS ================= */

.stTextInput input,
.stNumberInput input,
.stSelectbox div{
    border-radius:12px !important;
}


/* ================= QUICK BOX ================= */

.quick-box{
    background:white;
    padding:20px;
    border-radius:18px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ================= ADVANCED HELPERS (ML RULE ENGINE) =================

def get_stage(data):

    hba1c = data.get("HbA1c", data.get("HbAIc", 0))
    fbs = data.get("FBS", 0)
    gtt = data.get("GTT", data.get("GTT2Hr", 0))

    if hba1c >= 6.5 or fbs >= 126 or gtt >= 200:
        return "Diabetes Mellitus"

    elif (5.7 <= hba1c < 6.5) or (100 <= fbs < 126) or (140 <= gtt < 200):
        return "Prediabetes"

    return "Normal"


def get_type(data):

    # Type 1 logic (autoimmune markers)
    if data.get("AntiGad", 0) == 1 or data.get("IA2A", 0) == 1 or data.get("ICA", 0) == 1:
        return "Type 1 Diabetes"

    # Gestational
    if data.get("Pregnant", 0) == 1:
        return "Gestational Diabetes"

    # Secondary
    if data.get("Creatinine", 0) > 1.5 or data.get("Urea", 0) > 40:
        return "Secondary Diabetes"

    # Type 2 fallback
    if data.get("BMI", 0) > 30 and data.get("Age", 0) > 35:
        return "Type 2 Diabetes"

    return "Type 2 Diabetes"


def analyze_values(data):

    report = []

    if data.get("FBS", 0) >= 126:
        report.append(("FBS", "High", "Poor glucose control"))

    elif data.get("FBS", 0) < 70:
        report.append(("FBS", "Low", "Hypoglycemia risk"))

    if data.get("HbA1c", data.get("HbAIc", 0)) >= 6.5:
        report.append(("HbA1c", "High", "Diabetes range"))

    if data.get("LDL", 0) > 160:
        report.append(("LDL", "High", "Heart risk"))

    if data.get("HDL", 0) < 40:
        report.append(("HDL", "Low", "Poor heart protection"))

    if data.get("Triglycerides", 0) > 200:
        report.append(("Triglycerides", "High", "Metabolic risk"))

    if data.get("Creatinine", 0) > 1.3:
        report.append(("Creatinine", "High", "Kidney stress"))

    if data.get("EGFR", 100) < 60:
        report.append(("EGFR", "Low", "Kidney issue"))

    if data.get("SittingSystolicBP", 0) > 140:
        report.append(("Blood Pressure", "High", "Hypertension"))

    if data.get("SittingDiastolicBP", 0) > 90:
        report.append(("Blood Pressure", "High", "Hypertension"))

    return report


def predict_complications(data):

    risks = []

    if data.get("HbA1c", data.get("HbAIc", 0)) > 7:
        risks.append("Neuropathy")

    if data.get("Creatinine", 0) > 1.5:
        risks.append("Nephropathy")

    if data.get("LDL", 0) > 160:
        risks.append("Cardiovascular Disease")

    if data.get("FBS", 0) > 180:
        risks.append("Retinopathy")

    return risks


def give_recommendations(issues):

    recommendations = []

    if not issues:
        return ["Maintain healthy lifestyle"]

    for issue in issues:

        name = issue[0]

        if name in ["FBS", "HbA1c", "HbAIc"]:
            recommendations.append("Control blood sugar with diet & exercise")

        elif name in ["LDL", "Triglycerides"]:
            recommendations.append("Reduce oily and fast foods")

        elif name == "HDL":
            recommendations.append("Increase physical activity")

        elif name in ["Creatinine", "EGFR"]:
            recommendations.append("Monitor kidney function regularly")

        elif name == "Blood Pressure":
            recommendations.append("Control blood pressure")

    return list(set(recommendations))

def build_ml_input(data, model):
    df = pd.DataFrame([data])

    # sirf model ke features rakho
    for col in model.feature_names_in_:
        if col not in df:
            df[col] = 0

    df = df[model.feature_names_in_]

    # convert ALL to numeric (VERY IMPORTANT)
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    return df
    # ================= ML STEP 1 (DIABETES DETECTION) =================

def ml_predict_stage(data):
    df = build_ml_input(data, model1)

    df = df.reindex(columns=model1.feature_names_in_, fill_value=0)

    prob = model1.predict_proba(df)[0][1]

    if prob >= 0.7:
        return "Diabetes Mellitus"
    elif prob >= 0.4:
        return "Prediabetes"
    else:
        return "Normal"


# ================= ML STEP 2 (TYPE PREDICTION) =================
def ml_predict_type(data):

    df = pd.DataFrame([data])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    df = df.reindex(columns=model2.feature_names_in_, fill_value=0)

    return model2.predict(df)[0]

    # ================= FIX: CATEGORICAL TO NUMERIC =================
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # ================= SAFE NUMERIC CONVERSION =================
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ================= ALIGN FEATURES =================
    for col in model2.feature_names_in_:
        if col not in df:
            df[col] = 0

    df = df[model2.feature_names_in_]

    pred = model2.predict(df)[0]

    # ================= RULE OVERLAY =================
    if data.get("Pregnant", 0) == 1:
        return "Gestational Diabetes"

    if data.get("AntiGad", 0) == 1 or data.get("IA2A", 0) == 1:
        return "Type 1 Diabetes"

    return pred

# ================= ML STEP 3 (COMPLICATIONS) =================
def ml_predict_complications(data):

    df = pd.DataFrame([data])

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    df = df.reindex(columns=model3.feature_names_in_, fill_value=0)

    results = {}

    targets = ["Retinopathy","Nephropathy","Neuropathy","Cardiovascular","FootUlcer"]

    for i, est in enumerate(model3.estimators_):
        prob = est.predict_proba(df)[0][1]
        results[targets[i]] = {
            "probability": round(prob * 100, 2),
            "prediction": int(prob >= 0.55)
        }

    return results

    # ================= FIX 1: CATEGORICAL ENCODING =================
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    if "Pregnant" in df.columns:
        df["Pregnant"] = df["Pregnant"].astype(int)

    if "Smoker" in df.columns:
        df["Smoker"] = df["Smoker"].astype(int)

    if "AntiGad" in df.columns:
        df["AntiGad"] = df["AntiGad"].astype(int)

    if "IA2A" in df.columns:
        df["IA2A"] = df["IA2A"].astype(int)

    if "ICA" in df.columns:
        df["ICA"] = df["ICA"].astype(int)

    # ================= SAFE NUMERIC CONVERSION =================
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # ================= ALIGN FEATURES =================
    feature_cols = model3.feature_names_in_
    df = df.reindex(columns=feature_cols, fill_value=0)

    results = {}

    threshold = 0.55

    targets = [
        "Retinopathy",
        "Nephropathy",
        "Neuropathy",
        "Cardiovascular",
        "FootUlcer"
    ]

    for i, est in enumerate(model3.estimators_):

        prob = est.predict_proba(df)[0][1]

        results[targets[i]] = {
            "probability": round(prob * 100, 2),
            "prediction": int(prob >= threshold)
        }

    return results

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div class="logo-box">
    <div class="logo-title">🏥 DiabetesCare</div>
    <div class="logo-sub">AI Prediction System</div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    [
        "📊 Dashboard",
        "👨‍⚕️ Patients",
        "🧠 New Prediction",
        "📄 Reports",
        "📈 Analytics",
        "📅 Appointments"
    ]
)

# ================= DASHBOARD =================
if menu == "📊 Dashboard":

    left,right = st.columns([6,1])

    with left:

        st.markdown(
            '<div class="main-title">Dashboard</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="sub-title">Welcome, Dr. Ahmed Khan</div>',
            unsafe_allow_html=True
        )

    with right:

        current_date = datetime.now().strftime("%d %b %Y")

        st.markdown(
            f"""
            <div class="date-card">
                📅<br>
                <b>{current_date}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ================= COUNTS =================

    total_patients = len(st.session_state.patients)

    positive = len([
        p for p in st.session_state.patients
        if p["Stage"] == "Diabetes Mellitus"
    ])

    negative = len([
        p for p in st.session_state.patients
        if p["Stage"] == "Normal"
    ])

    risk = len([
        p for p in st.session_state.patients
        if p["Stage"] == "Prediabetes"
    ])

    reports = total_patients

    # ================= METRIC CARDS =================

    c1,c2,c3,c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">👨‍⚕️ Total Patients</div>
                <div class="metric-value" style="color:#0D9488;">{total_patients}</div>
                <div class="metric-growth">↑ Active Records</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🔵 Predictions Today</div>
                <div class="metric-value" style="color:#2563EB;">{positive + negative + risk}</div>
                <div class="metric-growth">↑ Today's Analysis</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🔴 Positive Cases</div>
                <div class="metric-value" style="color:#EF4444;">{positive}</div>
                <div class="metric-growth">↑ High Risk</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">🟣 Reports Generated</div>
                <div class="metric-value" style="color:#9333EA;">{reports}</div>
                <div class="metric-growth">↑ PDF Ready</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ================= TABLE + CHART =================

    col1,col2 = st.columns([1.7,1])

    with col1:

        st.markdown("### Recent Patients")

        if st.session_state.patients:

            df = pd.DataFrame(st.session_state.patients)

                # SAFE FIX FOR OLD DATA
            if "Type" not in df.columns:
                    df["Type"] = "N/A"

            if "Stage" not in df.columns:
                    df["Stage"] = "N/A"

            st.dataframe(
                df[[
                    "ID",
                    "Name",
                    "Age",
                    "Gender",
                    "Stage",
                    "Type"
                ]],
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No Patients Added Yet")

    with col2:

        st.markdown("### Prediction Overview")

        fig = go.Figure(data=[go.Pie(
            labels=["Positive","Negative","Prediabetes"],
            values=[positive,negative,risk],
            hole=.70
        )])

        fig.update_layout(
            height=340,
            paper_bgcolor="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ================= NEW PREDICTION =================
# ================= NEW PREDICTION =================
elif menu == "🧠 New Prediction":

    # ================= SESSION STEP =================

    if "prediction_step" not in st.session_state:
        st.session_state.prediction_step = 1

    step = st.session_state.prediction_step

    # ================= EXTRA CSS =================

    st.markdown("""
    <style>

    .prediction-card{
        background:white;
        padding:10px;
        border-radius:20px;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
        border-top:5px solid #1565C0;
        margin-bottom:40px;
    }

    .section-title{
        font-size:28px;
        font-weight:700;
        color:#263238;
        margin-bottom:5px;
    }

    .section-sub{
        color:#64748B;
        margin-bottom:25px;
    }

    .step-wrap{
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:35px;
    }

    .step-box{
        text-align:center;
        flex:1;
    }

    .step-circle{
        width:45px;
        height:45px;
        border-radius:50%;
        display:flex;
        align-items:center;
        justify-content:center;
        margin:auto;
        color:white;
        font-weight:bold;
        font-size:18px;
    }

    .active-step{
        background:#1565C0;
    }

    .inactive-step{
        background:#B0BEC5;
    }

    .complete-step{
        background:#2E7D32;
    }

    .step-text{
        margin-top:10px;
        font-size:15px;
        font-weight:600;
        color:#263238;
    }

    .step-subtext{
        font-size:13px;
        color:#64748B;
        margin-top:3px;
    }

    .line{
        height:4px;
        background:#D1D5DB;
        flex:1;
        margin:0 10px;
        margin-top:-25px;
    }

    .result-box{
        background:#F8FAFC;
        padding:18px;
        border-radius:16px;
        margin-top:12px;
        border-left:5px solid #1565C0;
    }
   
    </style>
    """, unsafe_allow_html=True)

    # ================= TITLE =================

    st.markdown(
    '<div class="main-title">New Prediction</div>',
    unsafe_allow_html=True
)

    st.markdown(
        f'<div class="sub-title">Step {st.session_state.prediction_step} of 3</div>',
        unsafe_allow_html=True
    )

    # ================= PROGRESS BAR =================

    c1, c2, c3, c4, c5 = st.columns([1,0.5,1,0.5,1])

    with c1:

        if step > 1:
            cls = "complete-step"
            icon = "✓"
        else:
            cls = "active-step"

            icon = "1"

        st.markdown(f"""
        <div class="step-box">
            <div class="step-circle {cls}">{icon}</div>
            <div class="step-text">Detect Diabetes</div>
            <div class="step-subtext">Patient Details</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    with c3:

        if step > 2:
            cls = "complete-step"
            icon = "✓"

        elif step == 2:
            cls = "active-step"
            icon = "2"

        else:
            cls = "inactive-step"
            icon = "2"

        st.markdown(f"""
        <div class="step-box">
            <div class="step-circle {cls}">{icon}</div>
            <div class="step-text">Etiological Type</div>
            <div class="step-subtext">Patient Details</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="line"></div>', unsafe_allow_html=True)

    with c5:

        if step == 3:
            cls = "active-step"
        else:
            cls = "inactive-step"

        st.markdown(f"""
        <div class="step-box">
            <div class="step-circle {cls}">3</div>
            <div class="step-text">Detect Complications</div>
            <div class="step-subtext">Patient Details</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # =========================================================
    # ================= STEP 1 =================
    # =========================================================
    

    if step == 1:

        st.markdown("""
        <div class="prediction-card">

        <div class="section-title">
        Patient Details
        </div>

        <div class="section-sub">
        Enter patient information for diabetes detection
        </div>

        </div>
        """, unsafe_allow_html=True)

        left,right = st.columns(2)

        with left:

            st.markdown("**Patient Name**")
            name = st.text_input("", placeholder="Enter full patient name", label_visibility="collapsed")

            st.markdown("**Age**")
            age = st.number_input("", min_value=1, placeholder="Example: 45", label_visibility="collapsed")

            st.markdown("**BMI**")
            bmi = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 28.50", label_visibility="collapsed")

            st.markdown("**Fasting Blood Sugar (FBS)**")
            fbs = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 126.00", label_visibility="collapsed")

            st.markdown("**HbA1c**")
            hba1c = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 6.80", label_visibility="collapsed")

            st.markdown("**Contact Number**")
            contact = st.text_input("", placeholder="03XX-XXXXXXX", label_visibility="collapsed")
        with right:
            st.markdown("**Gender**")
            gender = st.selectbox("", ["Male","Female"], label_visibility="collapsed")

            st.markdown("**Family History of Diabetes**")
            family_history = st.selectbox("", ["Yes","No"], label_visibility="collapsed")

            st.markdown("**2-Hour Glucose (GTT2Hr)**")
            gtt = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 180.00", label_visibility="collapsed")

            st.markdown("**LDL Cholesterol**")
            ldl = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 120.00", label_visibility="collapsed")

            st.markdown("**HDL Cholesterol**")
            hdl = st.number_input("", min_value=0.0, format="%.2f", placeholder="Example: 45.00", label_visibility="collapsed")

       

        # ================= STEP 1 BUTTON =================
        # ================= STEP 1 BUTTON =================
        if st.button("Next Step →"):

            if name.strip() == "" or age <= 0 or bmi <= 0 or fbs <= 0 or hba1c <= 0:

                st.error("⚠ Please fill complete patient information")

            else:

                # 🔥 FIX: encode categorical properly
                patient = {
                    "Age": age,
                    "Gender": 1 if gender == "Male" else 0,
                    "BMI": bmi,
                    "FBS": fbs,
                    "GTT2Hr": gtt,
                    "HbA1c": hba1c,
                    "LDL": ldl,
                    "HDL": hdl,
                    "FamilyHistory": 1 if family_history == "Yes" else 0
                }

                df = build_ml_input(patient, model1)

                # ================= STEP 1 LOGIC (FIXED) =================

                fbs_val = fbs
                hba1c_val = hba1c
                gtt_val = gtt

                if (fbs_val >= 126) or (hba1c_val >= 6.5) or (gtt_val >= 200):
                    stage = "Diabetes Mellitus"

                elif (100 <= fbs_val < 126) or (5.7 <= hba1c_val < 6.5) or (140 <= gtt_val < 200):
                    stage = "Prediabetes"

                else:
                    stage = "Normal"   # 🔥 direct predict (NO double logic)

                patient["Stage"] = stage

                st.session_state.patient_data = patient
                st.session_state.stage = stage
                st.session_state.show_result = True


        # ================= RESULT DISPLAY =================
        # ================= RESULT DISPLAY =================
        if st.session_state.get("show_result", False):

            stage = st.session_state.stage

            # ================= RESULT LOGIC =================
            if stage == "Normal":

                color = "#10B981"
                bg = "#ECFDF5"

                title = "✅ NO DIABETES DETECTED"
                subtitle = "Patient is currently in normal glycemic range"

                recommendations = [
                    "Maintain healthy diet",
                    "Exercise regularly",
                    "Annual diabetes screening",
                    "Avoid excessive sugar intake"
                ]

            elif stage == "Prediabetes":

                color = "#F59E0B"
                bg = "#FFFBEB"

                title = "⚠ PREDIABETES DETECTED"
                subtitle = "Patient is at risk of developing diabetes"

                recommendations = [
                    "Reduce sugar intake",
                    "Daily walking recommended",
                    "Weight management required",
                    "Monitor glucose regularly"
                ]

            else:

                color = "#EF4444"
                bg = "#FEF2F2"

                title = "🚨 DIABETES MELLITUS DETECTED"
                subtitle = "Further testing required"

                recommendations = [
                    "Immediate doctor consultation",
                    "HbA1c monitoring required",
                    "Strict diabetic diet",
                    "Proceed to next step"
                ]

            # ================= CARD =================
            with st.container():

                st.markdown(
                    f"""
                    <div style="
                        background:{bg};
                        padding:5px;
                        border-radius:22px;
                        border-left:8px solid {color};
                        box-shadow:0 4px 18px rgba(0,0,0,0.08);
                        margin-top:20px;
                        margin-bottom:20px;
                    ">
                    """,
                    unsafe_allow_html=True
                )

                # TITLE
                st.markdown(
                    f"""
                    <h2 style="
                        color:{color};
                        margin-bottom:8px;
                        font-size:30px;
                        font-weight:800;
                    ">
                        {title}
                    </h2>
                    """,
                    unsafe_allow_html=True
                )

                # SUBTITLE
                st.markdown(
                    f"""
                    <div style="
                        color:#475569;
                        font-size:16px;
                        margin-bottom:20px;
                    ">
                        {subtitle}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # RECOMMENDATION TITLE
                st.markdown(
                    """
                    <h3 style="
                        color:#0F172A;
                        margin-bottom:15px;
                    ">
                        🩺 Recommendations
                    </h3>
                    """,
                    unsafe_allow_html=True
                )

                # RECOMMENDATIONS
                for rec in recommendations:

                    st.markdown(
                        f"""
                        <div style="
                            background:white;
                            padding:12px 16px;
                            border-radius:12px;
                            margin-bottom:10px;
                            border:1px solid #E2E8F0;
                            color:#334155;
                            font-weight:500;
                        ">
                            ✔ {rec}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)
                    
            # ================= STEP 2 =================
            if stage == "Diabetes Mellitus":

                if st.button("Continue to Step 2 ➡"):

                    st.session_state.prediction_step = 2
                    st.rerun()
    # =========================================================
    # ================= STEP 2 =================
    # =========================================================

    elif step == 2:

        st.markdown("""
        <div class="prediction-card">

        <div class="section-title">
        Patient Details
        </div>

        <div class="section-sub">
        Enter patient information for etiological type detection
        </div>

        </div>
        
        """, unsafe_allow_html=True)

        left,right = st.columns(2)

        with left:

            st.markdown("**Pregnant (0/1)**")
            pregnant = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="pregnant"
            )

            st.markdown("**History Of GDM (0/1)**")
            history_gdm = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="history_gdm"
            )

            st.markdown("**Insulin Total Units**")
            insulin = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="insulin"
            )

            st.markdown("**Triglycerides**")
            triglycerides = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="triglycerides"
            )

            st.markdown("**DM Duration**")
            dm_duration = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="dm_duration"
            )

            st.markdown("**DM Onset Age**")
            onset_age = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="onset_age"
            )

        with right:

            st.markdown("**AntiGad (0/1)**")
            antigad = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="antigad"
            )

            st.markdown("**IA2A (0/1)**")
            ia2a = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="ia2a"
            )

            st.markdown("**ICA (0/1)**")
            ica = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="ica"
            )

            st.markdown("**Smoker (0/1)**")
            smoker = st.number_input(
                "",
                min_value=0.0,
                max_value=1.0,
                step=1.0,
                key="smoker"
            )

            st.markdown("**Urea**")
            urea = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="urea"
            )

            st.markdown("**Creatinine**")
            creatinine = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                key="creatinine"
    )
        st.info("⚠ Orange hint: Enter accurate lab values for better risk analysis")

        # ================= STEP 2 BUTTONS =================

        col_back, col_next = st.columns([1,1])

        with col_back:

            if st.button("⬅ Back"):

                st.session_state.prediction_step = 1
                st.rerun()

        with col_next:

            if st.button("Next ➡"):

                missing_fields = []

                if insulin <= 0:
                    missing_fields.append("Insulin")
                if triglycerides <= 0:
                    missing_fields.append("Triglycerides")
                if dm_duration <= 0:
                    missing_fields.append("DM Duration")
                if onset_age <= 0:
                    missing_fields.append("DM Onset Age")
                if urea <= 0:
                    missing_fields.append("Urea")
                if creatinine <= 0:
                    missing_fields.append("Creatinine")

                if missing_fields:
                    st.error("⚠ Please fill all required fields: " + ", ".join(missing_fields))

                else:

                    st.session_state.patient_data.update({

                        "Pregnant": pregnant,
                        "HistoryOfGDM": history_gdm,
                        "InsulinTotalUnits": insulin,
                        "Triglycerides": triglycerides,
                        "DMDuration": dm_duration,
                        "DMonSetAge": onset_age,
                        "AntiGad": antigad,
                        "IA2A": ia2a,
                        "ICA": ica,
                        "Smoker": smoker,
                        "Urea": urea,
                        "Creatinine": creatinine
                    })

                    patient = st.session_state.patient_data
                    patient["Type"] = ml_predict_type(patient)

                    st.session_state.prediction_step = 3
                    st.rerun()

    # =========================================================
    # ================= STEP 3 =================
    # =========================================================

    elif step == 3:

        st.markdown("""
        <div class="prediction-card">

        <div class="section-title">
        Patient Details
        </div>

        <div class="section-sub">
        Enter patient information for detect complications
        </div>

        </div>
        """, unsafe_allow_html=True)

        patient = st.session_state.patient_data

        

        # ================= COLUMNS =================
        left, right = st.columns(2)

        # ================= LEFT PANEL =================
        with left:
            

            st.markdown("**EGFR**")
            egfr = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                placeholder="Example: 90.00"
            )

            st.markdown("**Systolic BP**")
            systolic = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                placeholder="Example: 130.00"
            )

            st.markdown("**Diastolic BP**")
            diastolic = st.number_input(
                "",
                min_value=0.0,
                format="%.2f",
                placeholder="Example: 85.00"
            )

                    # ✅ TOGGLE STATE
            if "show_summary" not in st.session_state:
                st.session_state.show_summary = False

            st.markdown("### 👤 Patient Summary")

            # ✅ ONE BUTTON ONLY
            if st.button("Show / Hide Summary"):
                st.session_state.show_summary = not st.session_state.show_summary
                st.rerun()

            # ✅ SHOW ONLY WHEN TRUE
            if st.session_state.show_summary:

                st.markdown(f"""
                <div class="result-box">

                <b>Name:</b> {patient['Name']}<br><br>
                <b>Age:</b> {patient['Age']}<br><br>
                <b>BMI:</b> {patient['BMI']}<br><br>
                <b>FBS:</b> {patient['FBS']}<br><br>
                <b>HbA1c:</b> {patient['HbA1c']}<br><br>
                <b>LDL:</b> {patient['LDL']}<br><br>
                <b>HDL:</b> {patient['HDL']}

                </div>
                """, unsafe_allow_html=True)
                st.write("")
                st.write("")

            if st.button("⬅ Back"):
                st.session_state.prediction_step = 2
                st.rerun()

       

           # ================= RIGHT PANEL =================
        with right:

            st.markdown("### 📊 Prediction Preview")

            # ================= GENERATE FINAL PREDICTION =================

            import streamlit as st
            import streamlit.components.v1 as components

            
            if st.button("Generate Final Prediction"):

                missing_fields = []

                if egfr <= 0:
                    missing_fields.append("EGFR")
                if systolic <= 0:
                    missing_fields.append("Systolic BP")
                if diastolic <= 0:
                    missing_fields.append("Diastolic BP")

                if len(missing_fields) > 0:

                    st.error("⚠ Please fill: " + ", ".join(missing_fields))

                else:

                    # ================= GET DATA =================
                    patient = st.session_state.patient_data.copy()

                    patient["EGFR"] = egfr
                    patient["SystolicBP"] = systolic
                    patient["DiastolicBP"] = diastolic

                    
                    # ================= COMPPLICATIONS =================
                    comp_data = ml_predict_complications(patient)

                    complications = [
                        k for k, v in comp_data.items()
                        if v["prediction"] == 1
                    ]

                patient["Complications"] = ", ".join(complications) if complications else ""

                # ================= SAVE + DATABASE =================
                st.session_state.patient_data = patient
                add_patient(patient)
                st.session_state.patients = get_patients()

                # ================= ML PREDICTIONS =================
                stage = st.session_state.get("stage")   # STEP 1 stage
                dtype = patient.get("Type", "N/A")

                dtype = patient.get("Type", "N/A")

                # ================= RISK SCORE (FIXED) =================
                if stage == "Diabetes Mellitus":
                    risk_score = 85
                    result_text = "POSITIVE"
                    result_color = "#EF4444"
                    risk_text = "High Risk"

                elif stage == "Prediabetes":
                    risk_score = 55
                    result_text = "PRE-DIABETES"
                    result_color = "#F59E0B"
                    risk_text = "Moderate Risk"

                else:
                    risk_score = 20
                    result_text = "NEGATIVE"
                    result_color = "#10B981"
                    risk_text = "Low Risk"

                complications_text = patient.get("Complications") or "No major complications"

                # ================= RESPONSIVE HTML UI =================
                html_code = f"""
                <style>

                .wrapper {{
                    width:100%;
                    font-family: Arial;
                }}

                .result-main {{
                    background:white;
                    padding:20px;
                    border-radius:20px;
                    box-shadow:0 4px 14px rgba(0,0,0,0.08);
                    margin-top:10px;
                }}

                .result-grid {{
                    display:grid;
                    grid-template-columns:repeat(auto-fit, minmax(220px, 1fr));
                    gap:15px;
                }}

                .result-card {{
                    background:#F8FAFC;
                    padding:18px;
                    border-radius:16px;
                    border:1px solid #E2E8F0;
                }}

                .result-label {{
                    color:#64748B;
                    font-size:13px;
                    margin-bottom:8px;
                    font-weight:600;
                }}

                .result-big {{
                    font-size:24px;
                    font-weight:800;
                }}

                .result-sub {{
                    margin-top:8px;
                    font-size:13px;
                    color:#374151;
                }}

                .risk-value {{
                    font-size:36px;
                    font-weight:800;
                    color:#111827;
                }}

                .risk-bar {{
                    height:10px;
                    background:#E5E7EB;
                    border-radius:20px;
                    overflow:hidden;
                    margin-top:10px;
                }}

                .risk-fill {{
                    height:100%;
                    width:{risk_score}%;
                    background:#EF4444;
                }}

                ul {{
                    padding-left:18px;
                }}

                li {{
                    margin-bottom:6px;
                    font-size:13px;
                }}

                </style>

                <div class="wrapper">
                <div class="result-main">

                    <div class="result-grid">

                        <div class="result-card">
                            <div class="result-label">Diabetes Prediction</div>
                            <div class="result-big" style="color:{result_color};">
                                {result_text}
                            </div>
                            <div class="result-sub">{risk_text}</div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Type</div>
                            <div class="result-big" style="font-size:20px;color:#2563EB;">
                                {dtype}
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Stage</div>
                            <div class="result-big" style="font-size:20px;color:#F59E0B;">
                                {stage}
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Risk Score</div>
                            <div class="risk-value">{risk_score}%</div>
                            <div class="risk-bar">
                                <div class="risk-fill"></div>
                            </div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Complications</div>
                            <div class="result-sub">🩺 {complications_text}</div>
                        </div>

                        <div class="result-card">
                            <div class="result-label">Recommendations</div>
                            <ul>
                                <li>Healthy diet</li>
                                <li>Regular exercise</li>
                                <li>Monitor glucose</li>
                                <li>Avoid sugar</li>
                                <li>Doctor checkup</li>
                            </ul>
                        </div>

                    </div>

                </div>
                </div>
                """

                components.html(html_code, height=700, scrolling=False)

                st.success("Prediction Generated Successfully ✔")
                
# ================= PATIENTS =================
elif menu == "👨‍⚕️ Patients":

    import math
    import pandas as pd

    # ================= INIT STATE =================
    if "patients" not in st.session_state:
        st.session_state.patients = []

    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False

    # ================= CSS =================
    st.markdown("""
    <style>
    .patient-title{font-size:34px;font-weight:700;color:#111827;}
    .patient-sub{color:#6B7280;margin-top:-5px;}
    .patient-table{
        background:white;padding:20px;border-radius:18px;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
    }
    .table-header{font-weight:700;color:#374151;padding:10px 0;}
    .table-row{padding:10px 0;border-bottom:1px solid #F1F5F9;}
    .status-positive{color:#EF4444;font-weight:600;}
    .status-negative{color:#14B8A6;font-weight:600;}
    .status-risk{color:#F59E0B;font-weight:600;}
    </style>
    """, unsafe_allow_html=True)

    # ================= HEADER =================
    left, right = st.columns([6,2])

    with left:
        st.markdown("""
        <div class="patient-title">Patients</div>
        <div class="patient-sub">Manage all patients</div>
        """, unsafe_allow_html=True)

    with right:
        if st.button("➕ Add Patient", use_container_width=True):
            st.session_state.show_add_form = not st.session_state.show_add_form

    # ================= ADD PATIENT FORM =================
    if st.session_state.show_add_form:

        st.markdown("### ➕ Add New Patient")

        name = st.text_input("Name")
        age = st.number_input("Age", min_value=1, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        contact = st.text_input("Contact")

        if st.button("Save Patient"):

            if name.strip() == "":
                st.warning("Name required")
            else:
                new_patient = {
                    "ID": f"P{len(st.session_state.patients)+1}",
                    "Name": name,
                    "Age": age,
                    "Gender": gender,
                    "Contact": contact,
                    "Stage": "Normal",
                    "Date": datetime.now().strftime("%Y-%m-%d")
                }

                st.session_state.patients.append(new_patient)
                st.success("Patient Added ✔")
                st.session_state.show_add_form = False
                st.rerun()

    # ================= DATA CLEAN =================
    df = pd.DataFrame(st.session_state.patients)

    if df.empty:
        st.info("No patients found")
        st.stop()

    # remove NAN problem
    df = df.fillna("N/A")

    # safe columns
    if "Date" not in df.columns:
        df["Date"] = "N/A"

    # status
    def status_map(stage):
        if stage == "Diabetes Mellitus":
            return "Positive"
        elif stage == "Prediabetes":
            return "At Risk"
        return "Negative"

    df["Status"] = df["Stage"].apply(status_map)

    # ================= FILTERS =================
    c1, c2, c3 = st.columns([3,2,2])

    search = c1.text_input("🔍 Search patient")
    status_filter = c2.selectbox("Status", ["All", "Positive", "Negative", "At Risk"])
    gender_filter = c3.selectbox("Gender", ["All", "Male", "Female"])

    filtered = df.copy()

    if search:
        filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]

    if status_filter != "All":
        filtered = filtered[filtered["Status"] == status_filter]

    if gender_filter != "All":
        filtered = filtered[filtered["Gender"] == gender_filter]

    # ================= DELETE FUNCTION =================
        from database import delete_patient

        def delete_patient_local(pid):
            delete_patient(pid)   # DB se delete
            st.session_state.patients = get_patients() 

    # ================= PAGINATION =================
    ROWS = 8
    total_pages = max(1, math.ceil(len(filtered) / ROWS))

    if st.session_state.current_page > total_pages:
        st.session_state.current_page = total_pages

    start = (st.session_state.current_page - 1) * ROWS
    end = start + ROWS

    page_df = filtered.iloc[start:end]

    # ================= TABLE =================
    st.markdown('<div class="patient-table">', unsafe_allow_html=True)

    headers = ["ID","Name","Age","Gender","Contact","Date","Status","Action"]
    cols = st.columns(len(headers))

    for c, h in zip(cols, headers):
        c.markdown(f"**{h}**")

    for _, row in page_df.iterrows():

        c = st.columns(len(headers))

        c[0].write(row["ID"])
        c[1].write(row["Name"])
        c[2].write(row["Age"])
        c[3].write(row["Gender"])
        c[4].write(row["Contact"])
        c[5].write(row["Date"])

        color = "status-negative"
        if row["Status"] == "Positive":
            color = "status-positive"
        elif row["Status"] == "At Risk":
            color = "status-risk"

        c[6].markdown(f"<span class='{color}'>{row['Status']}</span>", unsafe_allow_html=True)

        # UNIQUE KEY FIX (IMPORTANT)
        btn_key = f"del_{row['ID']}_{row.name}"

        if c[7].button("🗑", key=btn_key):
            delete_patient(row["ID"])   # DB delete
            st.session_state.patients = get_patients()  # refresh
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # ================= PAGINATION UI =================
    p1, p2, p3 = st.columns([1,2,1])

    with p1:
        if st.button("⬅ Prev"):
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
                st.rerun()

    with p2:
        st.markdown(f"<center><b>Page {st.session_state.current_page} of {total_pages}</b></center>", unsafe_allow_html=True)

    with p3:
        if st.button("Next ➡"):
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
                st.rerun()
# ================= REPORTS =================
elif menu == "📄 Reports":

    st.markdown(
        '<div class="main-title">Reports</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">View and manage all generated reports</div>',
        unsafe_allow_html=True
    )

    # ================= DATA =================

    if st.session_state.patients:

        df = pd.DataFrame(st.session_state.patients)

        # SAFE COLUMNS
        if "Type" not in df.columns:
            df["Type"] = "-"

        if "Stage" not in df.columns:
            df["Stage"] = "Normal"

        if "Date" not in df.columns:
            df["Date"] = datetime.now().strftime("%d %b %Y")

        # ================= SEARCH + FILTER =================

        c1, c2 = st.columns([5,1])

        with c1:
            search = st.text_input(
                "",
                placeholder="🔍 Search reports...",
                label_visibility="collapsed"
            )

        with c2:
            filter_status = st.selectbox(
                "",
                ["All","Positive","Negative","At Risk"],
                label_visibility="collapsed"
            )

        # ================= STATUS MAP =================

        def report_status(stage):

            if stage == "Diabetes Mellitus":
                return "Positive"

            elif stage == "Prediabetes":
                return "At Risk"

            return "Negative"

        df["Result"] = df["Stage"].apply(report_status)

        # ================= SEARCH FILTER =================

        if search:
            df = df[
                df["Name"].str.contains(
                    search,
                    case=False,
                    na=False
                )
            ]

        if filter_status != "All":
            df = df[df["Result"] == filter_status]

        # ================= TABLE HEADER =================

        st.markdown("""
        <style>

        .report-table{
            background:white;
            padding:20px;
            border-radius:18px;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);
        }

        .positive{
            color:#EF4444;
            font-weight:700;
        }

        .negative{
            color:#10B981;
            font-weight:700;
        }

        .risk{
            color:#F59E0B;
            font-weight:700;
        }

        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="report-table">', unsafe_allow_html=True)

        headers = [
            "Report ID",
            "Patient Name",
            "Result",
            "Type",
            "Stage",
            "Date",
            "Action"
        ]

        cols = st.columns(len(headers))

        for c, h in zip(cols, headers):
            c.markdown(f"**{h}**")

        # ================= TABLE ROWS =================

        for index, row in df.iterrows():

            cols = st.columns(len(headers))

            cols[0].write(f"R{index+1:03}")
            cols[1].write(row["Name"])

            color = "negative"

            if row["Result"] == "Positive":
                color = "positive"

            elif row["Result"] == "At Risk":
                color = "risk"

            cols[2].markdown(
                f"<span class='{color}'>{row['Result']}</span>",
                unsafe_allow_html=True
            )

            cols[3].write(row["Type"])
            cols[4].write(row["Stage"])
            cols[5].write(row["Date"])

            if cols[6].button("⬇", key=f"download_{index}"):

                report_text = f'''
Patient Report
-------------------------

Patient Name: {row["Name"]}
Age: {row["Age"]}
Gender: {row["Gender"]}

Result: {row["Result"]}
Stage: {row["Stage"]}
Type: {row["Type"]}

Generated Date: {row["Date"]}
'''

                st.download_button(
                    label="Download",
                    data=report_text,
                    file_name=f"{row['Name']}_report.txt",
                    mime="text/plain",
                    key=f"dl_{index}"
                )

        st.markdown("</div>", unsafe_allow_html=True)

    else:

        st.info("No reports available")
# ================= ANALYTICS =================
# ================= ANALYTICS =================
elif menu == "📈 Analytics":

    st.markdown(
        '<div class="main-title">Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Insights and analytics overview</div>',
        unsafe_allow_html=True
    )

    if st.session_state.patients:

        df = pd.DataFrame(st.session_state.patients)

        # ================= COUNTS =================

        total_patients = len(df)

        positive = len(df[df["Stage"] == "Diabetes Mellitus"])

        negative = len(df[df["Stage"] == "Normal"])

        risk = len(df[df["Stage"] == "Prediabetes"])

        # ================= TOP CARDS =================

        c1,c2,c3,c4 = st.columns(4)

        with c1:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Total Patients</div>
                <div class="metric-value" style="color:#2563EB;">
                {total_patients}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Positive Cases</div>
                <div class="metric-value" style="color:#EF4444;">
                {positive}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">At Risk</div>
                <div class="metric-value" style="color:#F59E0B;">
                {risk}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c4:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Negative Cases</div>
                <div class="metric-value" style="color:#0D9488;">
                {negative}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        # ================= CHARTS =================

        left,right = st.columns(2)

        # ================= LINE CHART =================

        with left:

            st.markdown("### Prediction Trend")

            trend_df = pd.DataFrame({

                "Day":[
                    "1 May",
                    "4 May",
                    "7 May",
                    "10 May",
                    "13 May"
                ],

                "Positive":[
                    20,
                    15,
                    25,
                    18,
                    positive
                ],

                "Negative":[
                    60,
                    80,
                    70,
                    75,
                    negative
                ],

                "At Risk":[
                    30,
                    50,
                    45,
                    40,
                    risk
                ]
            })

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=trend_df["Day"],
                y=trend_df["Positive"],
                mode='lines+markers',
                name='Positive'
            ))

            fig.add_trace(go.Scatter(
                x=trend_df["Day"],
                y=trend_df["Negative"],
                mode='lines+markers',
                name='Negative'
            ))

            fig.add_trace(go.Scatter(
                x=trend_df["Day"],
                y=trend_df["At Risk"],
                mode='lines+markers',
                name='At Risk'
            ))

            fig.update_layout(

                height=400,

                paper_bgcolor="white",

                plot_bgcolor="white",

                margin=dict(l=20,r=20,t=30,b=20)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # ================= DONUT CHART =================

        with right:

            st.markdown("### Gender Distribution")

            male = len(df[df["Gender"] == "Male"])

            female = len(df[df["Gender"] == "Female"])

            fig2 = go.Figure(data=[go.Pie(

                labels=["Male","Female"],

                values=[male,female],

                hole=.65

            )])

            fig2.update_layout(

                height=400,

                paper_bgcolor="white"

            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

    else:

        st.info("No analytics data available")

# ================= APPOINTMENTS =================
elif menu == "📅 Appointments":

    st.markdown(
        '<div class="main-title">Appointments</div>',
        unsafe_allow_html=True
    )

    appointment_df = pd.DataFrame({

        "Patient":[
            "John Doe",
            "Sarah Khan",
            "Ali Raza"
        ],

        "Doctor":[
            "Dr. Ahmed",
            "Dr. Ahmed",
            "Dr. Ahmed"
        ],

        "Date":[
            "15 May",
            "16 May",
            "17 May"
        ],

        "Status":[
            "Confirmed",
            "Pending",
            "Confirmed"
        ]
    })

    st.dataframe(
        appointment_df,
        use_container_width=True,
        hide_index=True
    )
