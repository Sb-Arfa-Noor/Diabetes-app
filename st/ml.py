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

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="DiabetesCare AI Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

create_table()

# ================= SESSION STATE FOR MANAGEMENT =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signin"  # Options: 'signin' or 'signup'

# Secure dictionary for dynamic registration mapping
if "user_credentials" not in st.session_state:
    st.session_state.user_credentials = {
        "admin": "password"  # Default clinical master user
    }

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= PREMIUM ENTERPRISE CSS UI STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Page Overrides */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Strict Professional Navigation Header */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #94A3B8 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Radio Custom Selection Overlays */
    div[data-testid="stRadio"] label {
        padding: 10px 14px !important;
        border-radius: 6px !important;
        margin-bottom: 4px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    
    /* Authentication Panel Layout Matrix */
    .auth-wrapper {
        background: #FFFFFF;
        padding: 45px;
        border-radius: 12px;
        box-shadow: 0 4px 25px rgba(15, 23, 42, 0.04);
        border: 1px solid #E2E8F0;
        margin-top: 40px;
    }
    .auth-title-text {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .auth-subtitle-text {
        font-size: 13px;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
        margin-top: 4px;
    }
    
    /* Corporate KPI Summary Metrics Cards */
    .kpi-container {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.01);
    }
    .kpi-label-text {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value-text {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    /* Block Platform Containers */
    .custom-container {
        background: #FFFFFF;
        padding: 28px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Structural Section Header Layouts */
    .main-title {
        font-size: 30px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 25px;
        margin-top: 2px;
    }
    
    /* Global Button Theme Customization */
    .stButton>button {
        background-color: #0F766E !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 8px 20px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        border: none !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #115E59 !important;
        transform: translateY(-0.5px);
    }
    
    /* Form Section Dividers */
    .form-section-title {
        font-size: 16px;
        font-weight: 600;
        color: #1E293B;
        padding-bottom: 10px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ================= PROFESSIONAL AUTHENTICATION GATEWAY =================
if not st.session_state.logged_in:
    # Hide navigation panel until authentication is verified
    st.markdown("""<style>section[data-testid="stSidebar"] {display: none !important;}</style>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    left_gutter, center_content, right_gutter = st.columns([1.1, 1.2, 1.1])
    
    with center_content:
        # SIGN IN LAYER
        if st.session_state.auth_mode == "signin":
            st.markdown("""
                <div class='auth-wrapper'>
                    <div class='auth-title-text'>Clinical Portal Access</div>
                    <div class='auth-subtitle-text'>Provide validated clinical security keys to initialize dashboard nodes.</div>
                </div>
            """, unsafe_allow_html=True)
            
            input_username = st.text_input("Username Identifier", placeholder="Enter clinical username", key="login_user")
            input_password = st.text_input("Password Security Key", type="password", placeholder="Enter secure account password", key="login_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Authenticate Session", use_container_width=True):
                if input_username in st.session_state.user_credentials and st.session_state.user_credentials[input_username] == input_password:
                    st.session_state.logged_in = True
                    st.success("Session verified successfully.")
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid identity combination.")
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:25px;'>Unauthorized medical officer?</p>", unsafe_allow_html=True)
            if st.button("Register New Account", use_container_width=True):
                st.session_state.auth_mode = "signup"
                st.rerun()
                
        # SIGN UP LAYER
        elif st.session_state.auth_mode == "signup":
            st.markdown("""
                <div class='auth-wrapper'>
                    <div class='auth-title-text'>Practitioner Registry</div>
                    <div class='auth-subtitle-text'>Create new administrative user credentials for secure platform logging.</div>
                </div>
            """, unsafe_allow_html=True)
            
            reg_username = st.text_input("Select Unique Username", placeholder="e.g. dr_ahmed", key="reg_user")
            reg_password = st.text_input("Configure Strong Password", type="password", placeholder="Minimum 6 characters", key="reg_pass")
            reg_confirm  = st.text_input("Confirm Chosen Password", type="password", placeholder="Verify configuration match", key="reg_conf")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Complete Registration", use_container_width=True):
                if not reg_username.strip() or not reg_password.strip():
                    st.error("All credential initialization parameters are required.")
                elif reg_username in st.session_state.user_credentials:
                    st.error("This username key is already logged within memory arrays.")
                elif reg_password != reg_confirm:
                    st.error("Verification mismatch. Passwords do not correlate.")
                else:
                    st.session_state.user_credentials[reg_username] = reg_password
                    st.success("Account successfully structured. Please return to login module.")
                    st.session_state.auth_mode = "signin"
                    st.rerun()
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:25px;'>Existing practitioner?</p>", unsafe_allow_html=True)
            if st.button("Back to System Access Link", use_container_width=True):
                st.session_state.auth_mode = "signin"
                st.rerun()
    st.stop()

# ================= SIDEBAR NAVIGATION SCHEME =================
st.sidebar.markdown("""
<div style='padding: 20px 8px 10px 8px;'>
    <h2 style='color: #F8FAFC; font-weight: 700; font-size:22px; margin-bottom: 0px; letter-spacing:-0.5px;'>DiabetesCare AI</h2>
    <p style='color: #64748B; font-size: 12px; margin-top: 2px; font-weight:500;'>Clinical Management Unit</p>
</div>
<hr style='border-color: #1E293B; margin-top: 0px; margin-bottom:15px;'>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Hierarchy",
    ["Dashboard Overview", "Patients Database", "Diagnostic Pipeline", "Document Export Control", "Statistical Laboratory", "Allocation Scheduling"]
)

st.sidebar.markdown("<br><br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
if st.sidebar.button("Terminate Session", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.auth_mode = "signin"
    st.rerun()

# ================= MENU 1: DASHBOARD OVERVIEW =================
if menu == "Dashboard Overview":
    st.markdown('<div class="main-title">System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Overview of current registered clinical database records</div>', unsafe_allow_html=True)
    
    # Calculate Database Values
    total = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])
    normal = total - positive - risk
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="kpi-container" style="border-top: 4px solid #0F766E;"><div class="kpi-label-text">Total Health Metrics Logs</div><div class="kpi-value-text">{total}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-container" style="border-top: 4px solid #DC2626;"><div class="kpi-label-text">Confirmed Diabetes Cases</div><div class="kpi-value-text">{positive}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="kpi-container" style="border-top: 4px solid #D97706;"><div class="kpi-label-text">Prediabetes Risk Profiles</div><div class="kpi-value-text">{risk}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="kpi-container" style="border-top: 4px solid #0D9488;"><div class="kpi-label-text">Unregulated Normal Status</div><div class="kpi-value-text">{normal}</div></div>', unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("#### Recent Case Transactions")
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients).fillna("N/A")
            st.dataframe(df[["ID", "Name", "Age", "Gender", "Stage"]], use_container_width=True, hide_index=True)
        else:
            st.info("No records available in local storage arrays.")
            
    with col2:
        st.markdown("#### Population Distribution Ratio")
        fig = go.Figure(data=[go.Pie(
            labels=["Diabetes", "Normal", "Prediabetes"],
            values=[positive, normal, risk],
            hole=.65,
            marker=dict(colors=['#DC2626', '#0D9488', '#D97706'])
        )])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=True, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ================= MENU 2: DIAGNOSTIC PIPELINE =================
elif menu == "Diagnostic Pipeline":
    if "step" not in st.session_state:
        st.session_state.step = 1
    
    st.markdown('<div class="main-title">AI Predictive Inference Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Multi-stage analytics system pipeline — <b>Active Phase Frame {st.session_state.step} of 3</b></div>', unsafe_allow_html=True)
    
    # STEP 1
    if st.session_state.step == 1:
        st.markdown("<div class='custom-container'><div class='form-section-title'>Phase 1: Basic Clinical Metrics and Demographics</div>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            name = st.text_input("Patient Full Identity Name", placeholder="Enter complete string name")
            age = st.number_input("Patient Biological Age Value", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Biological Sex Classification", ["Male", "Female"])
            bmi = st.number_input("Calculated Body Mass Index (BMI Ratio)", min_value=10.0, max_value=60.0, value=24.5)
        with r:
            fbs = st.number_input("Fasting Blood Sugar Volume (FBS) [mg/dL]", min_value=50.0, max_value=400.0, value=100.0)
            hba1c = st.number_input("Laboratory Glycated Hemoglobin (HbA1c %)", min_value=3.0, max_value=15.0, value=5.8)
            gtt = st.number_input("Postprandial 2-Hour Glucose Tolerance (GTT) [mg/dL]", min_value=50.0, max_value=500.0, value=130.0)
            contact = st.text_input("Emergency Core Contact String", placeholder="Enter digital phone sequences")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Initialize Phase Evaluation"):
            if not name.strip():
                st.error("Operational constraint: Patient identity token required.")
            else:
                stage = "Normal"
                if hba1c >= 6.5 or fbs >= 126 or gtt >= 200: stage = "Diabetes Mellitus"
                elif 5.7 <= hba1c < 6.5 or 100 <= fbs < 126 or 140 <= gtt < 200: stage = "Prediabetes"
                
                st.session_state.patient_data = {
                    "Name": name, "Age": age, "Gender": gender, "BMI": bmi, 
                    "FBS": fbs, "HbA1c": hba1c, "GTT2Hr": gtt, "Contact": contact, "Stage": stage
                }
                
                if stage == "Diabetes Mellitus":
                    st.session_state.step = 2
                else:
                    p = st.session_state.patient_data.copy()
                    p["ID"] = f"P{len(st.session_state.patients)+1}"
                    p["Type"] = "N/A"
                    p["Complications"] = "None"
                    p["Date"] = datetime.now().strftime("%Y-%m-%d")
                    add_patient(p)
                    st.session_state.patients = get_patients()
                    st.session_state.step = 3
                st.rerun()

    # STEP 2
    elif st.session_state.step == 2:
        st.markdown("<div class='custom-container'><div class='form-section-title'>Phase 2: Pancreatic Testing and Advanced Isolation Panels</div>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            pregnant = st.selectbox("Gestational State Pregnancy Configuration Map", [0, 1], help="0 = Negative, 1 = Positive")
            insulin = st.number_input("Exogenous Clinical Insulin Intake Level Units", value=0.0)
            duration = st.number_input("Historical Duration Metrics of Symptoms (Years)", value=0.0)
            urea = st.number_input("Serum Blood Urea Clearance Metric", value=25.0)
        with r:
            antigad = st.selectbox("Anti-Glutamic Acid Decarboxylase (Anti-GAD) Antibody Assay", [0, 1])
            ica = st.selectbox("Islet Cell Cytoplasmic Autoantibodies (ICA) Target", [0, 1])
            creatinine = st.number_input("Serum Creatinine Clearance Filtration Metric", value=0.8)
            smoker = st.selectbox("Tobacco Nicotine Proclivity Factor Index", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Return to Phase 1"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("Process Subtype Stratification"):
                st.session_state.patient_data.update({
                    "Pregnant": pregnant, "InsulinTotalUnits": insulin, "DMDuration": duration,
                    "Urea": urea, "AntiGad": antigad, "ICA": ica, "Creatinine": creatinine, "Smoker": smoker,
                    "HistoryOfGDM": 0, "Triglycerides": 150.0, "DMonSetAge": st.session_state.patient_data["Age"], "IA2A": 0
                })
                
                if antigad == 1 or ica == 1: dtype = "Type 1 Diabetes"
                elif pregnant == 1: dtype = "Gestational Diabetes"
                else: dtype = "Type 2 Diabetes"
                
                st.session_state.patient_data["Type"] = dtype
                st.session_state.step = 3
                st.rerun()

    # STEP 3
    elif st.session_state.step == 3:
        st.markdown("<div class='custom-container'><div class='form-section-title'>Phase 3: Pipeline Output Evaluation Report</div>", unsafe_allow_html=True)
        p = st.session_state.patient_data
        
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 24px; border-radius: 6px; border: 1px solid #E2E8F0;">
                <h4 style="color: #0F766E; margin-top: 0; font-weight:700;">Diagnostic Architecture Verification Log</h4>
                <hr style="margin: 12px 0; border-color:#E2E8F0;">
                <p style="margin: 6px 0; font-size:14px;"><b>Patient Core Frame:</b> {p.get('Name')}</p>
                <p style="margin: 6px 0; font-size:14px;"><b>Demographic Vector:</b> {p.get('Age')} Years old | {p.get('Gender')}</p>
                <p style="margin: 6px 0; font-size:14px;"><b>Calculated Glycemic Condition:</b> <span style="color: #DC2626; font-weight: 700;">{p.get('Stage')}</span></p>
                <p style="margin: 6px 0; font-size:14px;"><b>Subtype Vector Classification:</b> {p.get('Type', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if p.get("Stage") == "Diabetes Mellitus" and "ID" not in p:
            if st.button("Commit Diagnostic Struct to Database Frame", use_container_width=True):
                p["ID"] = f"P{len(st.session_state.patients)+1}"
                p["Complications"] = "Retinopathy Microvascular Strain Evaluated" if p.get("DMDuration", 0) > 5 else "No High Microvascular Matrix Structural Risks"
                p["Date"] = datetime.now().strftime("%Y-%m-%d")
                add_patient(p)
                st.session_state.patients = get_patients()
                st.success("Log data securely integrated within active cluster arrays.")
                
        if st.button("Reset Pipeline Session", use_container_width=True):
            st.session_state.step = 1
            st.session_state.patient_data = {}
            st.rerun()

# ================= MENU 3: PATIENTS DATABASE =================
elif menu == "Patients Database":
    st.markdown('<div class="main-title">Electronic Health Records Registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Query interfaces and removal operations for institutional patient files</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients).fillna("N/A")
        
        search = st.text_input("Filter Local Nodes by Patient Name Token String", value="")
        if search:
            df = df[df["Name"].str.contains(search, case=False, na=False)]
            
        st.dataframe(df[["ID", "Name", "Age", "Gender", "Contact", "Stage", "Type"]], use_container_width=True, hide_index=True)
        
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown("##### Administrative Operations Controller")
        del_id = st.text_input("Enter Target Unique Row ID Key to Purge (e.g. P1)", placeholder="Enter ID string...")
        if st.button("Purge Database Entry Row"):
            if del_id:
                delete_patient(del_id)
                st.session_state.patients = get_patients()
                st.success(f"Row allocation {del_id} cleared from tracking layers.")
                st.rerun()
    else:
        st.info("System storage status: Registry queues are empty.")

# ================= MENU 4: DOCUMENT EXPORT CONTROL =================
elif menu == "Document Export Control":
    st.markdown('<div class="main-title">Export Verification Logs</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Download plain text official documentation summaries</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        for idx, row in pd.DataFrame(st.session_state.patients).iterrows():
            c = st.columns([1, 3, 2, 2])
            c[0].write(f"**{row.get('ID', 'P')}**")
            c[1].write(row.get("Name"))
            c[2].write(row.get("Stage"))
            
            report_body = f"DIABETESCARE AI DIAGNOSTIC REPORT\n=================================\nID: {row.get('ID')}\nPatient Name: {row.get('Name')}\nDiagnosis Stage: {row.get('Stage')}\nClassification Sub-type: {row.get('Type', 'N/A')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d')}"
            c[3].download_button("Export Object File", data=report_body, file_name=f"EHR_Report_{row.get('Name')}.txt", key=f"btn_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No documents are available to convert into down-stream object parameters.")

# ================= MENU 5: STATISTICAL LABORATORY =================
elif menu == "Statistical Laboratory":
    st.markdown('<div class="main-title">Population Epidemiological Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Epidemiological variable dispersion tracking charts</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        
        l, r = st.columns(2)
        with l:
            st.write("###### Blood Sugar Index (FBS) vs Demographics Scatter Framework")
            fig1 = px.scatter(df, x="Age", y="FBS", color="Stage", color_discrete_sequence=['#DC2626', '#0D9488', '#D97706'], template="simple_white")
            fig1.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        with r:
            st.write("###### Gender Dispersal Composition Profile")
            m_count = len(df[df["Gender"] == "Male"])
            f_count = len(df[df["Gender"] == "Female"])
            fig2 = go.Figure(data=[go.Pie(labels=["Male Vector Group", "Female Vector Group"], values=[m_count, f_count], hole=0.55, marker=dict(colors=['#0F766E','#BE123C']))])
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Statistical engines are non-functional until local database frames populate arrays.")

# ================= MENU 6: ALLOCATION SCHEDULER =================
elif menu == "Allocation Scheduling":
    st.markdown('<div class="main-title">Clinical Allocation Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Active time tracking and administrative consultation blocks</div>', unsafe_allow_html=True)
    
    sche_df = pd.DataFrame({
        "Patient Identification Index": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Assigned Healthcare Officer": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Time Allotment Allocation": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Approval Queue State": ["Approved State", "In Review Queue Frame", "Approved State"]
    })
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.dataframe(sche_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
