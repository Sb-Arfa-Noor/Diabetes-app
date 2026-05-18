# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
import math
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
    page_title="DiabetesCare AI - Clinical Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

create_table()

# ================= SESSION STATE INIT =================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= HIGH-END MEDICAL CORPORATE UI STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base Reset & Corporate Typography */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Professional Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        box-shadow: 1px 0 10px rgba(0,0,0,0.05);
    }
    section[data-testid="stSidebar"] * {
        color: #94A3B8 !important;
    }
    
    /* Clean Radio Buttons for Navigation */
    div[data-testid="stRadio"] label {
        padding: 10px 14px !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    
    /* Main Headings Configuration */
    .portal-header {
        font-size: 28px;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin-bottom: 2px;
    }
    .portal-subtitle {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 24px;
    }
    
    /* Decent Analytical Cards */
    .dashboard-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .card-meta {
        color: #64748B;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .card-value {
        font-size: 28px;
        font-weight: 700;
        margin-top: 4px;
        color: #1E293B;
    }
    .card-status-text {
        font-size: 12px;
        font-weight: 500;
        margin-top: 6px;
    }
    
    /* Wizard Steps Architecture Navigation Line */
    .step-container { display: flex; align-items: center; justify-content: space-between; width: 100%; margin-bottom: 20px; }
    .step-node { text-align: center; flex: 1; }
    .step-indicator {
        width: 32px; height: 32px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin: auto; font-weight: 600; font-size: 13px; transition: all 0.2s;
    }
    .step-active { background: #0284C7; color: white; }
    .step-inactive { background: #E2E8F0; color: #94A3B8; }
    .step-done { background: #0F766E; color: white; }
    .step-label { margin-top: 6px; font-size: 13px; font-weight: 500; color: #475569; }
    .step-divider { height: 2px; background: #E2E8F0; flex: 1; margin: 0 8px; margin-top: -22px; }

    /* Core Container Form Design */
    .content-panel {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .panel-header {
        font-size: 16px;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 8px;
    }
    
    /* Professional Alert Panels */
    .clinical-alert-banner {
        padding: 16px;
        border-radius: 6px;
        margin: 16px 0;
        border: 1px solid;
    }
    
    /* Decent System Buttons override */
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #0369A1;
    }
    
    /* Minimalistic Table Design Row */
    .table-row-container {
        display: flex;
        align-items: center;
        padding: 12px;
        border-bottom: 1px solid #F1F5F9;
    }
    .table-row-container:last-child {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# ================= CLEAN SECURE LOGIN GATE =================
if not st.session_state.authenticated:
    st.markdown("""
        <div style='text-align: center; padding-top: 80px;'>
            <h1 style='color: #0F172A; font-size: 32px; font-weight: 700; letter-spacing:-0.5px;'>Clinical Decision Portal</h1>
            <p style='color: #64748B; font-size: 14px; margin-top:4px;'>Authorized Medical Personnel Access Only</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1.2, 1.2, 1.2])
    with col2:
        st.markdown("<div class='content-panel' style='margin-top: 16px;'>", unsafe_allow_html=True)
        username = st.text_input("User Identification")
        password = st.text_input("Security Passcode", type="password")
        
        if st.button("Authenticate Account", use_container_width=True):
            if username == "admin" and password == "doctor123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Authentication Failure: Invalid medical credentials.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ================= BACKEND EXPERT ALGORITHMS =================
def get_stage(data):
    hba1c = data.get("HbA1c", data.get("HbAIc", 0))
    fbs = data.get("FBS", 0)
    gtt = data.get("GTT", data.get("GTT2Hr", 0))
    if hba1c >= 6.5 or fbs >= 126 or gtt >= 200: return "Diabetes Mellitus"
    elif (5.7 <= hba1c < 6.5) or (100 <= fbs < 126) or (140 <= gtt < 200): return "Prediabetes"
    return "Normal"

def get_type(data):
    if data.get("AntiGad", 0) == 1 or data.get("IA2A", 0) == 1 or data.get("ICA", 0) == 1: return "Type 1 Diabetes"
    if data.get("Pregnant", 0) == 1: return "Gestational Diabetes"
    if data.get("Creatinine", 0) > 1.5 or data.get("Urea", 0) > 40: return "Secondary Diabetes"
    if data.get("BMI", 0) > 30 and data.get("Age", 0) > 35: return "Type 2 Diabetes"
    return "Type 2 Diabetes"

def build_ml_input(data, model):
    df = pd.DataFrame([data])
    for col in model.feature_names_in_:
        if col not in df: df[col] = 0
    df = df[model.feature_names_in_]
    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

def ml_predict_stage(data):
    df = build_ml_input(data, model1)
    prob = model1.predict_proba(df)[0][1]
    if prob >= 0.7: return "Diabetes Mellitus"
    elif prob >= 0.4: return "Prediabetes"
    return "Normal"

def ml_predict_type(data):
    df = pd.DataFrame([data])
    if "Gender" in df.columns: df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.reindex(columns=model2.feature_names_in_, fill_value=0)
    return model2.predict(df)[0]

def ml_predict_complications(data):
    df = pd.DataFrame([data])
    if "Gender" in df.columns: df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    df = df.reindex(columns=model3.feature_names_in_, fill_value=0)
    results = {}
    targets = ["Retinopathy","Nephropathy","Neuropathy","Cardiovascular","FootUlcer"]
    for i, est in enumerate(model3.estimators_):
        prob = est.predict_proba(df)[0][1]
        results[targets[i]] = {"probability": round(prob * 100, 2), "prediction": int(prob >= 0.55)}
    return results

# ================= CORPORATE SIDEBAR SYSTEM =================
st.sidebar.markdown("""
<div style="padding: 16px 8px;">
    <div style="font-size:20px; font-weight:700; color:#F8FAFC; letter-spacing:-0.5px;">DiabetesCare AI</div>
    <div style="color:#64748B; font-size:11px; font-weight:600; letter-spacing:0.5px; margin-top:2px;">CLINICAL ENTERPRISE</div>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    ["Dashboard", "Patients", "New Prediction", "Reports", "Analytics", "Appointments"]
)

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
if st.sidebar.button("Log Out System", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

# ================= DASHBOARD MENU =================
if menu == "Dashboard":
    left, right = st.columns([6,2])
    with left:
        st.markdown('<div class="portal-header">Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="portal-subtitle">System Status: Active Clinical Registry Overview</div>', unsafe_allow_html=True)
    with right:
        current_date = datetime.now().strftime("%d %b %Y")
        st.markdown(f'<div style="background:white; padding:10px; border-radius:6px; text-align:center; border:1px solid #E2E8F0; font-size:13px; font-weight:500; color:#334155; margin-top:8px;">{current_date}</div>', unsafe_allow_html=True)

    total_patients = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    negative = len([p for p in st.session_state.patients if p.get("Stage") == "Normal"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="dashboard-card"><div class="card-meta">Total Registries</div><div class="card-value" style="color:#0284C7;">{total_patients}</div><div class="card-status-text" style="color:#64748B;">Active Patients</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="dashboard-card"><div class="card-meta">Confirmed Positive</div><div class="card-value" style="color:#DC2626;">{positive}</div><div class="card-status-text" style="color:#DC2626;">High Risk Management</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="dashboard-card"><div class="card-meta">Prediabetes Risk</div><div class="card-value" style="color:#D97706;">{risk}</div><div class="card-status-text" style="color:#D97706;">Observation Protocol</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="dashboard-card"><div class="card-meta">Confirmed Normal</div><div class="card-value" style="color:#0D9488;">{negative}</div><div class="card-status-text" style="color:#0D9488;">Baseline Regulated</div></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.7, 1])
    with col1:
        st.markdown("<div style='font-size:16px; font-weight:600; color:#1E293B; margin-bottom:12px;'>Recent Case Registry</div>", unsafe_allow_html=True)
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients)
            for col in ["Type", "Stage", "ID", "Name", "Age", "Gender"]:
                if col not in df.columns: df[col] = "N/A"
            st.dataframe(df[["ID", "Name", "Age", "Gender", "Stage", "Type"]], use_container_width=True, hide_index=True)
        else:
            st.info("No medical records registered yet.")
            
    with col2:
        st.markdown("<div style='font-size:16px; font-weight:600; color:#1E293B; margin-bottom:12px;'>Diagnostic Ratio Analysis</div>", unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=["Diabetes Positive", "Normal Baseline", "Prediabetes Risk"],
            values=[positive, negative, risk],
            hole=.7,
            marker=dict(colors=['#DC2626', '#0D9488', '#D97706'])
        )])
        fig.update_layout(
            height=260, 
            margin=dict(l=5, r=5, t=5, b=5), 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ================= CLINICAL PREDICTION COMPONENT =================
elif menu == "New Prediction":
    if "prediction_step" not in st.session_state: st.session_state.prediction_step = 1
    step = st.session_state.prediction_step

    st.markdown('<div class="portal-header">AI Engine Diagnostics</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="portal-subtitle">Execution Pipeline Terminal</div>', unsafe_allow_html=True)

    # Decent Minimal Multi-Stage Topbar
    c1, c2, c3, c4, c5 = st.columns([1, 0.4, 1, 0.4, 1])
    with c1:
        cls = "step-done" if step > 1 else "step-active"
        st.markdown(f'<div class="step-node"><div class="step-indicator {cls}">1</div><div class="step-label">Screening Assessment</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="step-divider"></div>', unsafe_allow_html=True)
    with c3:
        cls = "step-done" if step > 2 else ("step-active" if step == 2 else "step-inactive")
        st.markdown(f'<div class="step-node"><div class="step-indicator {cls}">2</div><div class="step-label">Etiological Isolation</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="step-divider"></div>', unsafe_allow_html=True)
    with c5:
        cls = "step-active" if step == 3 else "step-inactive"
        st.markdown(f'<div class="step-node"><div class="step-indicator {cls}">3</div><div class="step-label">Complication Matrix</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- STEP 1 ---
    if step == 1:
        st.markdown("<div class='content-panel'><div class='panel-header'>Patient Biometrics & Glycemic Level Parameters</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            name = st.text_input("Full Patient Name", placeholder="Identity string input")
            age = st.number_input("Age Parameter Profile", min_value=1, value=30)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, format="%.2f", value=22.0)
            fbs = st.number_input("Fasting Blood Sugar (FBS) [mg/dL]", min_value=0.0, format="%.2f", value=90.0)
            hba1c = st.number_input("HbA1c Lab Count (%)", min_value=0.0, format="%.2f", value=5.4)
            contact = st.text_input("Contact Identifier Data", placeholder="Telemetry sequence")
        with right:
            gender = st.selectbox("Biological Sex Profiling", ["Male", "Female"])
            family_history = st.selectbox("Genetic Diabetes Predisposition", ["No", "Yes"])
            gtt = st.number_input("2-Hour Glucose Tolerance Test (GTT) [mg/dL]", min_value=0.0, format="%.2f", value=120.0)
            ldl = st.number_input("LDL Serum Cholesterol [mg/dL]", min_value=0.0, format="%.2f", value=100.0)
            hdl = st.number_input("HDL Protective Cholesterol [mg/dL]", min_value=0.0, format="%.2f", value=50.0)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Proceed to Phase 2"):
            if not name.strip() or age <= 0 or bmi <= 0 or fbs <= 0 or hba1c <= 0:
                st.error("Execution Interrupted: Core biometrics cannot be evaluated at null values.")
            else:
                patient = {
                    "Name": name, "Age": age, "Gender": 1 if gender == "Male" else 0,
                    "BMI": bmi, "FBS": fbs, "GTT2Hr": gtt, "HbA1c": hba1c, "LDL": ldl, "HDL": hdl,
                    "FamilyHistory": 1 if family_history == "Yes" else 0, "Contact": contact
                }
                stage = "Diabetes Mellitus" if (fbs>=126 or hba1c>=6.5 or gtt>=200) else ("Prediabetes" if (100<=fbs<126 or 5.7<=hba1c<6.5 or 140<=gtt<200) else "Normal")
                patient["Stage"] = stage
                st.session_state.patient_data = patient
                st.session_state.stage = stage
                st.session_state.show_result = True

        if st.session_state.get("show_result", False):
            stage = st.session_state.stage
            color, bg, status_title = ("#DC2626", "#FEF2F2", "DIABETES MELLITUS PATHOLOGY LOCATED") if stage == "Diabetes Mellitus" else (("#D97706", "#FFFBEB", "PREDIABETES RISK VECTOR IDENTIFIED") if stage == "Prediabetes" else ("#0D9488", "#ECFDF5", "NEGATIVE CLINICAL DIAGNOSIS"))
            
            st.markdown(f"""
            <div class="clinical-alert-banner" style="background:{bg}; border-color:{color}; color: #1E293B;">
                <h5 style="color:{color}; margin-top:0; font-weight:600; margin-bottom:4px;">{status_title}</h5>
                <p style="font-size:13px; margin:0;">Target evaluation matrix matches the structural grouping criteria for: <b>{stage}</b>.</p>
            </div>
            """, unsafe_allow_html=True)

            if stage == "Diabetes Mellitus":
                if st.button("Initialize Etiological Diagnostics"):
                    st.session_state.prediction_step = 2
                    st.rerun()

    # --- STEP 2 ---
    elif step == 2:
        st.markdown("<div class='content-panel'><div class='panel-header'>Autoimmune Assays & Pancreatic Secretory Metrics</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            pregnant = st.number_input("Gestational Evaluation State (0 or 1)", min_value=0.0, max_value=1.0, step=1.0)
            history_gdm = st.number_input("Historical GDM Evaluation Sequence (0 or 1)", min_value=0.0, max_value=1.0, step=1.0)
            insulin = st.number_input("Exogenous Clinical Insulin Unit Volume", min_value=0.0, format="%.2f")
            triglycerides = st.number_input("Serum Triglyceride Content [mg/dL]", min_value=0.0, format="%.2f")
            dm_duration = st.number_input("Pathological Chronological Duration (Years)", min_value=0.0, format="%.2f")
            onset_age = st.number_input("Primary Diagnostic Onset Age", min_value=0.0, format="%.2f")
        with right:
            antigad = st.number_input("Anti-GAD Autoantibody Screening Index (0/1)", min_value=0.0, max_value=1.0, step=1.0)
            ia2a = st.number_input("IA-2A Reactive Marker Profiling Index (0/1)", min_value=0.0, max_value=1.0, step=1.0)
            ica = st.number_input("Islet Cell Cytoplasmic Antibody (ICA) Index (0/1)", min_value=0.0, max_value=1.0, step=1.0)
            smoker = st.number_input("Tobacco Exposure Factor Vector (0/1)", min_value=0.0, max_value=1.0, step=1.0)
            urea = st.number_input("Serum Urea Measurements [mg/dL]", min_value=0.0, format="%.2f")
            creatinine = st.number_input("Serum Creatinine Nephrology Indicator [mg/dL]", min_value=0.0, format="%.2f")
        st.markdown("</div>", unsafe_allow_html=True)

        cb, cn = st.columns(2)
        with cb:
            if st.button("Previous Phase"):
                st.session_state.prediction_step = 1
                st.rerun()
        with cn:
            if st.button("Process Isolation Models"):
                st.session_state.patient_data.update({
                    "Pregnant": pregnant, "HistoryOfGDM": history_gdm, "InsulinTotalUnits": insulin,
                    "Triglycerides": triglycerides, "DMDuration": dm_duration, "DMonSetAge": onset_age,
                    "AntiGad": antigad, "IA2A": ia2a, "ICA": ica, "Smoker": smoker, "Urea": urea, "Creatinine": creatinine
                })
                patient = st.session_state.patient_data
                patient["Type"] = ml_predict_type(patient)
                st.session_state.prediction_step = 3
                st.rerun()

    # --- STEP 3 ---
    elif step == 3:
        st.markdown("<div class='content-panel'><div class='panel-header'>Chronic Microvascular & Macrovascular Complications Modeling</div>", unsafe_allow_html=True)
        left, right = st.columns(2)
        with left:
            egfr = st.number_input("Estimated GFR (eGFR Filtration Clearance)", min_value=0.0, value=90.0)
            systolic = st.number_input("Resting Systolic Arterial Pressure", min_value=0.0, value=120.0)
            diastolic = st.number_input("Resting Diastolic Arterial Pressure", min_value=0.0, value=80.0)
            
            if "show_summary" not in st.session_state: st.session_state.show_summary = False
            if st.button("Toggle Summary Stream"):
                st.session_state.show_summary = not st.session_state.show_summary
                st.rerun()
            
            if st.session_state.show_summary:
                p = st.session_state.patient_data
                st.markdown(f"""
                <div style='background:#F1F5F9; padding:12px; border-radius:4px; font-size:13px; margin-top:8px; border:1px solid #E2E8F0; color:#334155;'>
                    <b>Identity:</b> {p.get('Name')} | <b>Age:</b> {p.get('Age')} | <b>BMI:</b> {p.get('BMI')}<br>
                    <b>FBS Parameter:</b> {p.get('FBS')} mg/dL | <b>HbA1c Lab Count:</b> {p.get('HbA1c')}%
                </div>
                """, unsafe_allow_html=True)
            
            st.write("<br>", unsafe_allow_html=True)
            if st.button("Return to Type Processing"):
                st.session_state.prediction_step = 2
                st.rerun()

        with right:
            if st.button("Generate Diagnostic Synthesis Report", use_container_width=True):
                patient = st.session_state.patient_data.copy()
                patient["EGFR"], patient["SystolicBP"], patient["DiastolicBP"] = egfr, systolic, diastolic
                
                comp_data = ml_predict_complications(patient)
                complications = [k for k, v in comp_data.items() if v["prediction"] == 1]
                patient["Complications"] = ", ".join(complications) if complications else "No Active Risk Factors"
                
                stage = ml_predict_stage(patient)
                patient["Stage"] = stage
                dtype = patient.get("Type", "Type 2 Diabetes")
                
                risk_score = 85 if stage == "Diabetes Mellitus" else (55 if stage == "Prediabetes" else 20)
                res_text = "DIABETES DETECTED" if stage == "Diabetes Mellitus" else ( "PREDIABETES RISK STATUS" if stage == "Prediabetes" else "NEGATIVE LAB STATUS")
                res_color = "#DC2626" if stage == "Diabetes Mellitus" else ("#D97706" if stage == "Prediabetes" else "#0D9488")
                
                patient["ID"] = f"P{len(st.session_state.patients)+1}"
                patient["Date"] = datetime.now().strftime("%Y-%m-%d")
                add_patient(patient)
                st.session_state.patients = get_patients()
                
                st.markdown(f"""
                <div style='background: white; border:1px solid #E2E8F0; padding:16px; border-radius:6px; margin-top:10px;'>
                    <div style='font-size:11px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.5px;'>Pipeline Evaluation Synthesis</div>
                    <div style='font-size:22px; font-weight:700; color:{res_color}; margin-top:2px;'>{res_text}</div>
                    <hr style='margin:10px 0; border-color:#F1F5F9;'>
                    <p style='font-size:13px; margin: 4px 0;'><b>Diagnostic Target Class:</b> {stage}</p>
                    <p style='font-size:13px; margin: 4px 0;'><b>Identified Etiology Strain:</b> {dtype}</p>
                    <p style='font-size:13px; margin: 4px 0;'><b>Complications Risk Matrix:</b> <span style='color:#DC2626; font-weight:500;'>{patient["Complications"]}</span></p>
                    <p style='font-size:13px; margin: 4px 0;'><b>Quantized Probability Index:</b> {risk_score}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.success("Electronic Medical Registry Document Synchronized.")
        st.markdown("</div>", unsafe_allow_html=True)

# ================= PATIENTS MANAGEMENT MENU =================
elif menu == "Patients":
    st.markdown('<div class="portal-header">Patient Registry Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="portal-subtitle">EHR Data Storage Query Matrix</div>', unsafe_allow_html=True)

    if "current_page" not in st.session_state: st.session_state.current_page = 1
    if "show_add_form" not in st.session_state: st.session_state.show_add_form = False

    if st.button("Register New Walk-in Admission Profile"):
        st.session_state.show_add_form = not st.session_state.show_add_form

    if st.session_state.show_add_form:
        st.markdown("<div class='content-panel' style='margin-top:12px;'>", unsafe_allow_html=True)
        n = st.text_input("Name string token")
        a = st.number_input("Age value discrete", min_value=1, step=1)
        g = st.selectbox("Biological identification marker", ["Male", "Female"])
        c = st.text_input("Contact numeric telemetry")
        
        if st.button("Commit Profile to Structural Database"):
            if not n.strip(): st.warning("Execution Halt: Name value cannot remain null.")
            else:
                new_p = {
                    "ID": f"P{len(st.session_state.patients)+1}", "Name": n, "Age": a, "Gender": g,
                    "Contact": c, "Stage": "Normal", "Type": "N/A", "Date": datetime.now().strftime("%Y-%m-%d")
                }
                add_patient(new_p)
                st.session_state.patients = get_patients()
                st.success("EHR Profile Created.")
                st.session_state.show_add_form = False
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    df = pd.DataFrame(st.session_state.patients)
    if df.empty:
        st.info("System database holds no current archived matrices.")
        st.stop()

    df = df.fillna("N/A")
    if "Date" not in df.columns: df["Date"] = "N/A"
    df["Status"] = df["Stage"].apply(lambda s: "Positive" if s=="Diabetes Mellitus" else ("At Risk" if s=="Prediabetes" else "Negative"))

    f1, f2 = st.columns(2)
    search = f1.text_input("Search Patient Matrix Identity Keys")
    status_filter = f2.selectbox("Isolate System Diagnostic Class", ["All", "Positive", "Negative", "At Risk"])

    filtered = df.copy()
    if search: filtered = filtered[filtered["Name"].str.contains(search, case=False, na=False)]
    if status_filter != "All": filtered = filtered[filtered["Status"] == status_filter]

    ROWS = 6
    total_pages = max(1, math.ceil(len(filtered) / ROWS))
    if st.session_state.current_page > total_pages: st.session_state.current_page = total_pages
    start = (st.session_state.current_page - 1) * ROWS
    page_df = filtered.iloc[start:start+ROWS]

    st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
    for index, row in page_df.iterrows():
        c = st.columns([1, 2, 1, 1, 2, 2, 1])
        c[0].write(f"**{row['ID']}**")
        c[1].write(row["Name"])
        c[2].write(f"{row['Age']} yrs")
        c[3].write(row["Gender"])
        c[4].write(row["Contact"])
        
        lbl_clr = "#DC2626" if row["Status"]=="Positive" else ("#D97706" if row["Status"]=="At Risk" else "#0D9488")
        c[5].markdown(f"<span style='color:{lbl_clr}; font-weight:600; font-size:13px;'>{row['Status']}</span>", unsafe_allow_html=True)
        
        if c[6].button("Delete", key=f"del_{row['ID']}_{index}"):
            delete_patient(row["ID"])
            st.session_state.patients = get_patients()
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    p1, p2, p3 = st.columns([1, 2, 1])
    with p1:
        if st.button("Previous Matrix") and st.session_state.current_page > 1:
            st.session_state.current_page -= 1
            st.rerun()
    with p2: st.markdown(f"<center><p style='font-size:12px; color:#64748B;'>Index Registry {st.session_state.current_page} of {total_pages}</p></center>", unsafe_allow_html=True)
    with p3:
        if st.button("Next Matrix") and st.session_state.current_page < total_pages:
            st.session_state.current_page += 1
            st.rerun()

# ================= CLINICAL REPORTS EXPORT MATRIX =================
elif menu == "Reports":
    st.markdown('<div class="portal-header">EHR Telemetry Data Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="portal-subtitle">Exportable clinical validation data units</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        for col in ["Type", "Stage", "Date"]:
            if col not in df.columns: df[col] = "N/A"

        st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
        for idx, row in df.iterrows():
            c = st.columns([1, 3, 2, 2, 2])
            c[0].write(f"**LOG-{idx+1:02d}**")
            c[1].write(row["Name"])
            c[2].write(row["Stage"])
            c[3].write(row["Date"])
            
            report_text = f"Patient Diagnosis Report\n===================\nName: {row['Name']}\nStage: {row['Stage']}\nType: {row['Type']}\nDate Verified: {row['Date']}\n"
            c[4].download_button("Export Telemetry", data=report_text, file_name=f"EHR_{row['Name']}.txt", mime="text/plain", key=f"dl_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No compiled text streams located on cloud environment servers.")

# ================= ANALYTICS SYSTEM METRICS =================
elif menu == "Analytics":
    st.markdown('<div class="portal-header">Statistical Epidemiology Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="portal-subtitle">Population baseline data analytics</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        total_patients = len(df)
        pos = len(df[df["Stage"] == "Diabetes Mellitus"])
        neg = len(df[df["Stage"] == "Normal"])
        risk = len(df[df["Stage"] == "Prediabetes"])

        l, r = st.columns(2)
        with l:
            st.markdown("<div style='font-size:14px; font-weight:600; margin-bottom:8px;'>Chronological Trend Metrics</div>", unsafe_allow_html=True)
            trend_df = pd.DataFrame({
                "Timeline": ["Sequence Alpha", "Sequence Beta", "Sequence Gamma", "Active Frame"],
                "Active Positives": [5, 12, 19, pos],
                "Baseline Clear": [20, 45, 62, neg]
            })
            fig = px.line(trend_df, x="Timeline", y=["Active Positives", "Baseline Clear"], color_discrete_sequence=['#DC2626', '#0D9488'], template="simple_white")
            fig.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with r:
            st.markdown("<div style='font-size:14px; font-weight:600; margin-bottom:8px;'>Biological Sex Variable Spread</div>", unsafe_allow_html=True)
            m = len(df[df["Gender"] == "Male"]) + len(df[df["Gender"] == 1])
            f = total_patients - m
            fig2 = go.Figure(data=[go.Pie(labels=["Male Segments", "Female Segments"], values=[m, f], hole=0.6, marker=dict(colors=['#0284C7','#E11D48']))])
            fig2.update_layout(height=240, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("System tracking graphs empty until input vectors populate structural arrays.")

# ================= APPOINTMENTS INTERFACE =================
elif menu == "Appointments":
    st.markdown('<div class="portal-header">Clinical Scheduler</div>', unsafe_allow_html=True)
    st.markdown('<div class="portal-subtitle">Time matrix alignment allocation</div>', unsafe_allow_html=True)

    appointment_df = pd.DataFrame({
        "Patient Identity Matrix": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Consulting Personnel": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Time Window": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Deployment State": ["Validated Unit", "Queue Phase", "Validated Unit"]
    })
    st.markdown("<div class='content-panel'>", unsafe_allow_html=True)
    st.dataframe(appointment_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
