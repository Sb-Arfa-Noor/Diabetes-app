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

# ================= SESSION STATE ARCHITECTURE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_screen" not in st.session_state:
    st.session_state.auth_screen = "login"  # 'login' or 'register'

# In-memory authentication database tracker
if "auth_registry" not in st.session_state:
    st.session_state.auth_registry = {
        "admin": "password"  # Core Static Master Admin Account
    }

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= PREMIUM NO-EMOJI DECENT CSS STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Container Theme */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Professional Login Layout Card */
    .auth-container-box {
        background: #FFFFFF;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 30px rgba(15, 23, 42, 0.04);
        border: 1px solid #E2E8F0;
        margin-top: 35px;
    }
    .auth-header-main {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .auth-header-sub {
        font-size: 13px;
        color: #64748B;
        text-align: center;
        margin-top: 6px;
        margin-bottom: 25px;
    }
    
    /* Corporate Dashboard Navigation Menu */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #94A3B8 !important;
    }
    
    /* Clean Form Layout Cards */
    .panel-card-container {
        background: #FFFFFF;
        padding: 25px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Text Typography Controls */
    .main-title-view {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
    .sub-title-view {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 25px;
    }
    
    /* High-End Metric Card Nodes */
    .metric-node-box {
        background: #FFFFFF;
        padding: 22px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .metric-node-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-node-value {
        font-size: 26px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    /* Unified Structural Professional Button Overrides */
    .stButton>button {
        background-color: #0F766E !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        border: none !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background-color: #115E59 !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= FUNCTIONAL AUTHENTICATION FLOW =================
if not st.session_state.logged_in:
    # Sidebar ko block kar dein jab tak login complete nahi hota
    st.markdown("""<style>section[data-testid="stSidebar"] {display: none !important;}</style>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    gutter_left, center_auth, gutter_right = st.columns([1.1, 1.2, 1.1])
    
    with center_auth:
        # MODULE A: SIGN IN FORM
        if st.session_state.auth_screen == "login":
            st.markdown("""
                <div class='auth-container-box'>
                    <div class='auth-header-main'>Clinical Portal Access</div>
                    <div class='auth-header-sub'>Enter administrative account keys or use admin default account</div>
                </div>
            """, unsafe_allow_html=True)
            
            login_user = st.text_input("Username", placeholder="e.g. admin", key="txt_login_user")
            login_pass = st.text_input("Password", type="password", placeholder="Enter account password", key="txt_login_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In to System", use_container_width=True):
                # Proper clean database value checking
                if login_user in st.session_state.auth_registry and st.session_state.auth_registry[login_user] == login_pass:
                    st.session_state.logged_in = True
                    st.success("Session verified.")
                    st.rerun()
                else:
                    st.error("Access denied. Invalid credentials parameter.")
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:20px;'>Need a separate clinical account?</p>", unsafe_allow_html=True)
            if st.button("Create Practitioner Account", use_container_width=True):
                st.session_state.auth_screen = "register"
                st.rerun()
                
        # MODULE B: SIGN UP FORM
        elif st.session_state.auth_screen == "register":
            st.markdown("""
                <div class='auth-container-box'>
                    <div class='auth-header-main'>Practitioner Registration</div>
                    <div class='auth-header-sub'>Structure new valid security credentials into active portal memory</div>
                </div>
            """, unsafe_allow_html=True)
            
            reg_user = st.text_input("Choose Username", placeholder="e.g. dr_fatima", key="txt_reg_user")
            reg_pass = st.text_input("Choose Password", type="password", placeholder="Set secure password", key="txt_reg_pass")
            reg_conf = st.text_input("Confirm Password", type="password", placeholder="Re-type password", key="txt_reg_conf")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register Account", use_container_width=True):
                if not reg_user.strip() or not reg_pass.strip():
                    st.error("Please fill all required operational entry structures.")
                elif reg_user in st.session_state.auth_registry:
                    st.error("This target user identification token is already logged.")
                elif reg_pass != reg_conf:
                    st.error("Configurations mismatch. Passwords must be perfectly equivalent.")
                else:
                    # Sync to application memory database array
                    st.session_state.auth_registry[reg_user] = reg_pass
                    st.success("Registration complete! You can now sign in using these keys.")
                    st.session_state.auth_screen = "login"
                    st.rerun()
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:20px;'>Already have access keys?</p>", unsafe_allow_html=True)
            if st.button("Return to Sign In Panel", use_container_width=True):
                st.session_state.auth_screen = "login"
                st.rerun()
    st.stop()

# ================= CORPORATE NAVIGATION MENU =================
st.sidebar.markdown("""
<div style='padding: 20px 8px 10px 8px;'>
    <div style='color: #F8FAFC; font-weight: 700; font-size:22px; letter-spacing:-0.5px;'>DiabetesCare AI</div>
    <div style='color: #64748B; font-size: 12px; margin-top: 2px; font-weight:500;'>Clinical Management Architecture</div>
</div>
<hr style='border-color: #1E293B; margin-top: 0px; margin-bottom:15px;'>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation System",
    ["Dashboard Overview", "Patients Matrix Registry", "Diagnostic Pipeline", "Data Report Center", "Visual Analytics Node", "Consultation Matrix"]
)

st.sidebar.markdown("<br><br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
if st.sidebar.button("Log Out System Workspace", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.auth_screen = "login"
    st.rerun()

# ================= DASHBOARD OVERVIEW =================
if menu == "Dashboard Overview":
    st.markdown('<div class="main-title-view">System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Overview of current registered clinical database records</div>', unsafe_allow_html=True)
    
    total = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])
    normal = total - positive - risk
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-node-box" style="border-top: 4px solid #0F766E;"><div class="metric-node-label">Total Logs Managed</div><div class="metric-node-value">{total}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-node-box" style="border-top: 4px solid #DC2626;"><div class="metric-node-label">Confirmed Positive Cases</div><div class="metric-node-value">{positive}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-node-box" style="border-top: 4px solid #D97706;"><div class="metric-node-label">Prediabetes Metrics Risk</div><div class="metric-node-value">{risk}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-node-box" style="border-top: 4px solid #0D9488;"><div class="metric-node-label">Normal Physiological Status</div><div class="metric-node-value">{normal}</div></div>', unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("#### Recent Case Profiles Integrated")
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients).fillna("N/A")
            st.dataframe(df[["ID", "Name", "Age", "Gender", "Stage"]], use_container_width=True, hide_index=True)
        else:
            st.info("No cases currently recorded within institutional structures.")
    with col2:
        st.markdown("#### Demographic Ratio Metrics")
        fig = go.Figure(data=[go.Pie(labels=["Diabetes State", "Normal State", "Prediabetes State"], values=[positive, normal, risk], hole=.65, marker=dict(colors=['#DC2626', '#0D9488', '#D97706']))])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=True, paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ================= DIAGNOSTIC PIPELINE =================
elif menu == "Diagnostic Pipeline":
    if "step" not in st.session_state:
        st.session_state.step = 1
    
    st.markdown('<div class="main-title-view">AI Inference Architecture</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title-view">Multi-stage analytics system pipeline — <b>Active Phase Frame {st.session_state.step} of 3</b></div>', unsafe_allow_html=True)
    
    if st.session_state.step == 1:
        st.markdown("<div class='panel-card-container'><h5>Phase 1: Entry Demographics & Direct Laboratory Glycemic Markers</h5><br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            name = st.text_input("Patient Full Identity Name Sequence", placeholder="Enter clinical patient text data name")
            age = st.number_input("Patient Age Value", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Biological Sex Taxonomy Configuration", ["Male", "Female"])
            bmi = st.number_input("Calculated Body Mass Index (BMI Value Ratio)", min_value=10.0, max_value=60.0, value=24.5)
        with r:
            fbs = st.number_input("Fasting Blood Sugar Volume (FBS) [mg/dL]", min_value=50.0, max_value=400.0, value=100.0)
            hba1c = st.number_input("Laboratory Glycated Hemoglobin (HbA1c Count Percent %)", min_value=3.0, max_value=15.0, value=5.8)
            gtt = st.number_input("2-Hour Postprandial Glucose Tolerance Validation (GTT) [mg/dL]", min_value=50.0, max_value=500.0, value=130.0)
            contact = st.text_input("Core Contact Mapping Value Sequence", placeholder="Enter numeric string digits")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Process Initial Metrics Framework"):
            if not name.strip():
                st.error("Constraint violated: Patient legal identity text vector must be structured.")
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

    elif st.session_state.step == 2:
        st.markdown("<div class='panel-card-container'><h5>Phase 2: Etiological Isolation Panels & Pancreatic Endocrine Assays</h5><br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            pregnant = st.selectbox("Gestational Pregnancy Status Framework", [0, 1])
            insulin = st.number_input("Exogenous Insulin Therapeutic Load Volume Units", value=0.0)
            duration = st.number_input("Historical Sensation Index Duration (Years)", value=0.0)
            urea = st.number_input("Serum Blood Urea Clearance Filtration Index", value=25.0)
        with r:
            antigad = st.selectbox("Anti-Glutamic Acid Decarboxylase (Anti-GAD) Antibody Matrix", [0, 1])
            ica = st.selectbox("Islet Cell Cytoplasmic Autoantibody (ICA) Matrix Verification", [0, 1])
            creatinine = st.number_input("Serum Creatinine Clear Level Metrics", value=0.8)
            smoker = st.selectbox("Tobacco Proclivity Intake Factor Index", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Return to Phase 1 Framework"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("Evaluate Secondary Stratification Subtype"):
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

    elif st.session_state.step == 3:
        st.markdown("<div class='panel-card-container'><h5>Phase 3: Pipeline Output Diagnostic Struct Analysis</h5>", unsafe_allow_html=True)
        p = st.session_state.patient_data
        
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 22px; border-radius: 6px; border: 1px solid #E2E8F0;">
                <h5 style="color: #0F766E; margin-top: 0; font-weight:700;">Diagnostic Engine Analytics Evaluation Result</h5>
                <hr style="margin: 12px 0; border-color:#E2E8F0;">
                <p style="margin: 4px 0; font-size:14px;"><b>Identity Label Reference:</b> {p.get('Name')}</p>
                <p style="margin: 4px 0; font-size:14px;"><b>Physiological Vectors:</b> {p.get('Age')} Years | {p.get('Gender')}</p>
                <p style="margin: 4px 0; font-size:14px;"><b>Glycemic Level State Map:</b> <span style="color: #DC2626; font-weight: 700;">{p.get('Stage')}</span></p>
                <p style="margin: 4px 0; font-size:14px;"><b>Isolated Secondary Strain Subtype:</b> {p.get('Type', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if p.get("Stage") == "Diabetes Mellitus" and "ID" not in p:
            if st.button("Commit Diagnostic Struct to Secure Database Rows", use_container_width=True):
                p["ID"] = f"P{len(st.session_state.patients)+1}"
                p["Complications"] = "Retinopathy Risk Tracking Target Verified" if p.get("DMDuration", 0) > 5 else "Clear Structural Matrix Boundaries"
                p["Date"] = datetime.now().strftime("%Y-%m-%d")
                add_patient(p)
                st.session_state.patients = get_patients()
                st.success("Log status successfully pushed to permanent cluster variables.")
                
        if st.button("Initialize Fresh Pipeline Tracking Framework", use_container_width=True):
            st.session_state.step = 1
            st.session_state.patient_data = {}
            st.rerun()

# ================= PATIENTS MATRIX REGISTRY =================
elif menu == "Patients Matrix Registry":
    st.markdown('<div class="main-title-view">Electronic Health Ledger Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Registry control layers and transactional data table frames</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients).fillna("N/A")
        search = st.text_input("Filter Local Framework Nodes by Identity Name String", value="")
        if search:
            df = df[df["Name"].str.contains(search, case=False, na=False)]
            
        st.dataframe(df[["ID", "Name", "Age", "Gender", "Contact", "Stage", "Type"]], use_container_width=True, hide_index=True)
        
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown("##### Administrative Operations Node Controller")
        del_target = st.text_input("Target Row Unique ID Reference Sequence to Drop (e.g. P1)", placeholder="Enter row identity mapping sequence...")
        if st.button("Purge Database Entity Frame"):
            if del_target:
                delete_patient(del_target)
                st.session_state.patients = get_patients()
                st.success(f"Row allocation {del_target} successfully unlinked from data architecture.")
                st.rerun()
    else:
        st.info("No compiled metrics logged inside storage records matrix lines.")

# ================= DATA REPORT CENTER =================
elif menu == "Data Report Center":
    st.markdown('<div class="main-title-view">Documentation Export Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Download formal plain-text electronic verification logs</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        st.markdown("<div class='panel-card-container'>", unsafe_allow_html=True)
        for idx, row in pd.DataFrame(st.session_state.patients).iterrows():
            c = st.columns([1, 3, 2, 2])
            c[0].write(f"**{row.get('ID', 'P')}**")
            c[1].write(row.get("Name"))
            c[2].write(row.get("Stage"))
            
            report_body = f"DIABETESCARE AI DIAGNOSTIC REPORT\n=================================\nID: {row.get('ID')}\nPatient Name: {row.get('Name')}\nDiagnosis Stage: {row.get('Stage')}\nClassification Sub-type: {row.get('Type', 'N/A')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d')}"
            c[3].download_button("Export Plain Text Log", data=report_body, file_name=f"EHR_Report_{row.get('Name')}.txt", key=f"btn_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No file object generation arrays detected in data nodes.")

# ================= VISUAL ANALYTICS NODE =================
elif menu == "Visual Analytics Node":
    st.markdown('<div class="main-title-view">Statistical Laboratory Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Epidemiological variable mapping layouts</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        l, r = st.columns(2)
        with l:
            st.write("###### Fasting Blood Sugar (FBS) vs Patient Age Distribution Graph")
            fig1 = px.scatter(df, x="Age", y="FBS", color="Stage", color_discrete_sequence=['#DC2626', '#0D9488', '#D97706'], template="simple_white")
            fig1.update_layout(paper_bgcolor="white")
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        with r:
            st.write("###### Sex Categorization Matrix Breakdown Proportions")
            m_size = len(df[df["Gender"] == "Male"])
            f_size = len(df[df["Gender"] == "Female"])
            fig2 = go.Figure(data=[go.Pie(labels=["Male Proportions Cluster", "Female Proportions Cluster"], values=[m_size, f_size], hole=0.58, marker=dict(colors=['#0F766E','#BE123C']))])
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Laboratory graphs are frozen until system values populate memory arrays.")

# ================= CONSULTATION MATRIX =================
elif menu == "Consultation Matrix":
    st.markdown('<div class="main-title-view">Clinical Allocation Matrix Grid</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Time logs tracking operational consultation scheduling blocks</div>', unsafe_allow_html=True)
    
    sche_df = pd.DataFrame({
        "Patient Identity Mapping String": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Assigned Medical Practitioner": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Allocation Time Window": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Current Verification Pipeline Flags": ["Confirmed Structure Log", "In Review Validation Queue", "Confirmed Structure Log"]
    })
    st.markdown("<div class='panel-card-container'>", unsafe_allow_html=True)
    st.dataframe(sche_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
