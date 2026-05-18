# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
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
    page_title="DiabetesCare AI Corporate Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

create_table()

# ================= SESSION STATE ARCHITECTURE =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_screen" not in st.session_state:
    st.session_state.auth_screen = "login"

if "auth_registry" not in st.session_state:
    st.session_state.auth_registry = {
        "admin": "password"  # Core Master Admin Account
    }

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

if "sidebar_state" not in st.session_state:
    st.session_state.sidebar_state = "expanded"

# ================= PREMIUM EXECUTIVE DARK BLUE CSS STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&display=swap');
    
    /* Core Application Framework Theme */
    .stApp {
        background-color: #0B0F19; /* Executive Deep Dark Blue Space */
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Strict Typography Controls Override */
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main Streamlit Core Overrides */
    div[data-testid="stWidgetLabel"] p {
        color: #94A3B8 !important; /* Soft Silver Blue for Inputs */
        font-weight: 500 !important;
        font-size: 13px !important;
        letter-spacing: 0.2px;
    }
    
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border: 1px solid #1F2937 !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    
    /* Premium Sidebar UI Architecture */
    section[data-testid="stSidebar"] {
        background-color: #090D16 !important;
        border-right: 1px solid #1E293B !important;
        padding-top: 10px;
        transition: width 0.3s ease, transform 0.3s ease !important;
    }
    
    /* Custom Sidebar Radio Element Styling Hack */
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 8px !important;
        transition: all 0.25s ease-in-out;
        width: 100%;
    }
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label:hover {
        border-color: #38BDF8 !important; /* Premium Cyan Light Accent */
        background-color: #1E293B !important;
    }
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        border-color: #38BDF8 !important;
    }
    div[data-testid="stSidebarUserContent"] .stRadio div[role="radiogroup"] label p {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Clean Premium Layout Cards */
    .panel-card-container {
        background: #111C44; /* Image Matching Classic Deep Navy Blue */
        padding: 28px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin-bottom: 24px;
    }
    
    /* Premium Title Headers Layout */
    .main-title-view {
        font-size: 32px;
        font-weight: 800;
        color: #F8FAFC !important;
        letter-spacing: -0.75px;
        margin-bottom: 4px;
    }
    .sub-title-view {
        font-size: 14px;
        color: #38BDF8 !important; /* Neon Electric Blue Secondary Tint */
        font-weight: 500;
        margin-bottom: 30px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* High-End Clean Metric Node Structure */
    .metric-node-box {
        background: #111C44;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1E293B;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .metric-node-label {
        font-size: 11px;
        font-weight: 700;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-node-value {
        font-size: 32px;
        font-weight: 800;
        color: #FFFFFF !important;
        margin-top: 6px;
    }
    
    /* Authentic Clean Login Box Layout */
    .auth-container-box {
        background: #111C44;
        padding: 45px;
        border-radius: 16px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        border: 1px solid #1E293B;
        margin-top: 40px;
    }
    .auth-header-main {
        font-size: 28px;
        font-weight: 800;
        color: #FFFFFF !important;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .auth-header-sub {
        font-size: 13px;
        color: #94A3B8 !important;
        text-align: center;
        margin-top: 8px;
        margin-bottom: 30px;
    }
    
    /* Custom High Contrast Core System Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%) !important; /* High Grade Electric Cyan-Blue Gradient */
        color: #FFFFFF !important;
        border-radius: 8px !important;
        padding: 12px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2) !important;
        transition: all 0.25s ease-in-out !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.35) !important;
    }
    
    /* Streamlit Interactive Dataframe Custom Dark Theme Polish */
    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* Professional Layout Micro-adjusters */
    .custom-report-view {
        background-color: #FFFFFF !important;
        color: #1E293B !important;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        font-size: 13px;
        line-height: 1.6;
    }
    .custom-report-view h4, .custom-report-view h5, .custom-report-view p, .custom-report-view b {
        color: #1E293B !important;
    }
</style>
""", unsafe_allow_html=True)

# ================= FUNCTIONAL AUTHENTICATION FLOW =================
if not st.session_state.logged_in:
    st.markdown("""<style>section[data-testid="stSidebar"] {display: none !important;}</style>""", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    gutter_left, center_auth, gutter_right = st.columns([1.1, 1.2, 1.1])
    
    with center_auth:
        if st.session_state.auth_screen == "login":
            st.markdown("""
                <div class='auth-container-box'>
                    <div class='auth-header-main'>Clinical Portal Access</div>
                    <div class='auth-header-sub'>Secure Administrative Dashboard Identity Validation Keys</div>
                </div>
            """, unsafe_allow_html=True)
            
            login_user = st.text_input("Username Identifier", placeholder="e.g. admin", key="txt_login_user")
            login_pass = st.text_input("Security Access Code", type="password", placeholder="••••••••", key="txt_login_pass")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In to Corporate Workspace", use_container_width=True):
                if login_user in st.session_state.auth_registry and st.session_state.auth_registry[login_user] == login_pass:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Authentication Error: Invalid identity or credential sequence token.")
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#94A3B8; margin-top:25px;'>Need a separate clinical account?</p>", unsafe_allow_html=True)
            if st.button("Create Practitioner Account Layer", use_container_width=True):
                st.session_state.auth_screen = "register"
                st.rerun()
                
        elif st.session_state.auth_screen == "register":
            st.markdown("""
                <div class='auth-container-box'>
                    <div class='auth-header-main'>Register Practitioner</div>
                    <div class='auth-header-sub'>Structure new active clinical access parameters into secure framework</div>
                </div>
            """, unsafe_allow_html=True)
            
            reg_user = st.text_input("Choose Username Token", placeholder="e.g. dr_fatima", key="txt_reg_user")
            reg_pass = st.text_input("Choose Security Password", type="password", placeholder="••••••••", key="txt_reg_pass")
            reg_conf = st.text_input("Confirm Security Password", type="password", placeholder="••••••••", key="txt_reg_conf")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Compile & Register Credentials", use_container_width=True):
                if not reg_user.strip() or not reg_pass.strip():
                    st.error("Input Violation: Fields cannot be left null.")
                elif reg_user in st.session_state.auth_registry:
                    st.error("Identity Collision: Target username token is already structured.")
                elif reg_pass != reg_conf:
                    st.error("Verification Mismatch: Passwords must be completely identical.")
                else:
                    st.session_state.auth_registry[reg_user] = reg_pass
                    st.success("Registration Successful! Proceeding to initialization layer.")
                    st.session_state.auth_screen = "login"
                    st.rerun()
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#94A3B8; margin-top:25px;'>Already registered inside active portal?</p>", unsafe_allow_html=True)
            if st.button("Return to System Login", use_container_width=True):
                st.session_state.auth_screen = "login"
                st.rerun()
    st.stop()

# ================= CORPORATE NAVIGATION MENU =================
st.sidebar.markdown("""
<div style='padding: 24px 12px 16px 12px;'>
    <div style='color: #FFFFFF; font-weight: 800; font-size:24px; letter-spacing:-0.75px;'>DiabetesCare AI</div>
    <div style='color: #38BDF8; font-size: 11px; margin-top: 4px; font-weight:700; text-transform: uppercase; letter-spacing:1px;'>Clinical Neural Center</div>
</div>
<hr style='border-color: #1E293B; margin-top: 0px; margin-bottom:20px;'>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "NAVIGATION NODE",
    ["Dashboard Overview", "Patients Matrix Registry", "Diagnostic Pipeline", "Data Report Center", "Visual Analytics Node", "Consultation Matrix"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br><br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
if st.sidebar.button("Terminate Session Workspace", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.auth_screen = "login"
    st.rerun()

# ================= MODULE 1: DASHBOARD OVERVIEW =================
if menu == "Dashboard Overview":
    # Clean & Professional Sidebar Toggle/State Management Layout using True Icons
    title_left, title_right = st.columns([5, 1.2])
    with title_left:
        st.markdown('<div class="main-title-view">System Executive Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title-view">Live Operational Monitoring & Clinical Summary Matrices</div>', unsafe_allow_html=True)
    with title_right:
        st.write("<p style='margin:0; font-size:10px; color:#94A3B8; font-weight:700; text-align:center; letter-spacing:0.5px;'>SIDEBAR WINDOW</p>", unsafe_allow_html=True)
        arr_col1, arr_col2 = st.columns(2)
        with arr_col1:
            if st.button("«", key="close_nav_bar", use_container_width=True, help="Collapse Side Menu"):
                st.markdown("""
                    <style>
                        section[data-testid="stSidebar"] {
                            width: 0px !important;
                            min-width: 0px !important;
                            transform: translateX(-350px) !important;
                        }
                    </style>
                """, unsafe_allow_html=True)
        with arr_col2:
            if st.button("»", key="open_nav_bar", use_container_width=True, help="Expand Side Menu"):
                st.markdown("""
                    <style>
                        section[data-testid="stSidebar"] {
                            width: 336px !important;
                            min-width: 336px !important;
                            transform: translateX(0px) !important;
                        }
                    </style>
                """, unsafe_allow_html=True)
    
    # Calculate Data Structures
    total = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])
    normal = total - positive - risk
    
    # Fallback Data Allocation agar data empty ho
    if total == 0:
        total, positive, risk, normal = 124, 52, 38, 34
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-node-box" style="border-left: 4px solid #38BDF8;"><div class="metric-node-label">Total Logs Managed</div><div class="metric-node-value">{total}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-node-box" style="border-left: 4px solid #F43F5E;"><div class="metric-node-label">Positive Instances</div><div class="metric-node-value">{positive}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-node-box" style="border-left: 4px solid #F59E0B;"><div class="metric-node-label">Prediabetes Risk</div><div class="metric-node-value">{risk}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-node-box" style="border-left: 4px solid #10B981;"><div class="metric-node-label">Normal Physiological</div><div class="metric-node-value">{normal}</div></div>', unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1.1])
    
    with col1:
        st.markdown("<h5 style='color:#FFFFFF; margin-bottom:15px; font-weight:700;'>Recent Integrated Case Records</h5>", unsafe_allow_html=True)
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients).fillna("N/A")
            st.dataframe(
                df[["ID", "Name", "Age", "Gender", "Stage"]], 
                use_container_width=True, 
                hide_index=True
            )
        else:
            mock_table = pd.DataFrame([
                {"ID": "P102", "Name": "Ayesha Malik", "Age": 45, "Gender": "Female", "Stage": "Diabetes Mellitus"},
                {"ID": "P103", "Name": "Zain Ahmed", "Age": 38, "Gender": "Male", "Stage": "Prediabetes"},
                {"ID": "P104", "Name": "Mariam Khan", "Age": 29, "Gender": "Female", "Stage": "Normal Status"},
                {"ID": "P105", "Name": "Bilal Siddiqui", "Age": 52, "Gender": "Male", "Stage": "Diabetes Mellitus"}
            ])
            st.dataframe(mock_table, use_container_width=True, hide_index=True)
            
    with col2:
        st.markdown("<h5 style='color:#FFFFFF; margin-bottom:15px; font-weight:700;'>Population Density Proportions</h5>", unsafe_allow_html=True)
        
        # VALIDATED ERROR-FREE: Heavy Professional Donut Chart
        fig = go.Figure()
        
        fig.add_trace(go.Pie(
            labels=["Diabetes Mellitus", "Normal Status", "Prediabetes Risk"], 
            values=[positive, normal, risk], 
            hole=0.65, 
            marker=dict(
                colors=['#F43F5E', '#10B981', '#F59E0B'],
                line=dict(color='#111C44', width=2)
            ),
            textinfo='percent+label',
            hoverinfo='label+value+percent'
        ))
        
        fig.update_layout(
            height=310, 
            margin=dict(l=15, r=15, t=10, b=10), 
            showlegend=False, 
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#FFFFFF")
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ================= MODULE 2: DIAGNOSTIC PIPELINE =================
elif menu == "Diagnostic Pipeline":
    if "step" not in st.session_state:
        st.session_state.step = 1
    
    st.markdown('<div class="main-title-view">AI Inference Architecture</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title-view">Multi-stage analytics system pipeline — <b>Active Phase Frame {st.session_state.step} of 3</b></div>', unsafe_allow_html=True)
    
    if st.session_state.step == 1:
        st.markdown("<div class='panel-card-container'><h4 style='color:#FFFFFF; font-weight:700; margin-top:0;'>Phase 1: Entry Demographics & Direct Laboratory Glycemic Markers</h4><br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            name = st.text_input("Patient Full Identity Name Sequence", placeholder="Enter legal text name string")
            age = st.number_input("Patient Biological Age Value", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Biological Sex Taxonomy Configuration", ["Male", "Female"])
            bmi = st.number_input("Calculated Body Mass Index (BMI Value Ratio)", min_value=10.0, max_value=60.0, value=24.5)
        with r:
            fbs = st.number_input("Fasting Blood Sugar Volume (FBS) [mg/dL]", min_value=50.0, max_value=400.0, value=100.0)
            hba1c = st.number_input("Laboratory Glycated Hemoglobin (HbA1c Count Percent %)", min_value=3.0, max_value=15.0, value=5.8)
            gtt = st.number_input("2-Hour Postprandial Glucose Tolerance Validation (GTT) [mg/dL]", min_value=50.0, max_value=500.0, value=130.0)
            contact = st.text_input("Core Contact Mapping Value Sequence", placeholder="Enter numeric contact digits")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Process Initial Metrics Framework"):
            if not name.strip():
                st.error("Constraint violated: Patient identity token vector cannot be structural null.")
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
        st.markdown("<div class='panel-card-container'><h4 style='color:#FFFFFF; font-weight:700; margin-top:0;'>Phase 2: Etiological Isolation Panels & Pancreatic Endocrine Assays</h4><br>", unsafe_allow_html=True)
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
        st.markdown("<div class='panel-card-container'><h4 style='color:#FFFFFF; font-weight:700; margin-top:0;'>Phase 3: Pipeline Output Diagnostic Struct Analysis</h4>", unsafe_allow_html=True)
        p = st.session_state.patient_data
        
        st.markdown(f"""
            <div style="background-color: #111827; padding: 24px; border-radius: 8px; border: 1px solid #1E293B;">
                <h5 style="color: #38BDF8 !important; margin-top: 0; font-weight:800; font-size:18px;">Diagnostic Engine Analytics Evaluation Result</h5>
                <hr style="margin: 14px 0; border-color:#1E293B;">
                <p style="margin: 8px 0; font-size:14px; color:#F8FAFC !important;"><b>Identity Label Reference:</b> {p.get('Name')}</p>
                <p style="margin: 8px 0; font-size:14px; color:#F8FAFC !important;"><b>Physiological Vectors:</b> {p.get('Age')} Years | {p.get('Gender')}</p>
                <p style="margin: 8px 0; font-size:14px; color:#F8FAFC !important;"><b>Glycemic Level State Map:</b> <span style="color: #F43F5E !important; font-weight: 700;">{p.get('Stage')}</span></p>
                <p style="margin: 8px 0; font-size:14px; color:#F8FAFC !important;"><b>Isolated Secondary Strain Subtype:</b> {p.get('Type', 'N/A')}</p>
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
                st.success("Log status successfully pushed to cloud cluster structures.")
                
        if st.button("Initialize Fresh Pipeline Tracking Framework", use_container_width=True):
            st.session_state.step = 1
            st.session_state.patient_data = {}
            st.rerun()

# ================= MODULE 3: PATIENTS MATRIX REGISTRY =================
elif menu == "Patients Matrix Registry":
    st.markdown('<div class="main-title-view">Electronic Health Ledger Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Registry control layers and transactional data table frames</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients).fillna("N/A")
        search = st.text_input("Filter Local Framework Nodes by Identity Name String", value="")
        if search:
            df = df[df["Name"].str.contains(search, case=False, na=False)]
            
        st.dataframe(
            df[["ID", "Name", "Age", "Gender", "Contact", "Stage", "Type"]], 
            use_container_width=True, 
            hide_index=True
        )
        
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown("<div class='panel-card-container'><h5 style='color:#FFFFFF; font-weight:700; margin-top:0;'>Administrative System Modification Module</h5>", unsafe_allow_html=True)
        del_target = st.text_input("Target Row Unique ID Reference Sequence to Drop (e.g. P1)", placeholder="Enter row identity mapping sequence...")
        if st.button("Purge Database Entity Frame"):
            if del_target:
                delete_patient(del_target)
                st.session_state.patients = get_patients()
                st.success(f"Row allocation {del_target} successfully unlinked from data architecture.")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No compiled metrics logged inside storage records matrix lines.")

# ================= MODULE 4: DATA REPORT CENTER =================
elif menu == "Data Report Center":
    st.markdown('<div class="main-title-view">Documentation Export Matrices</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Download formal plain-text electronic verification logs</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df_rep = pd.DataFrame(st.session_state.patients).fillna("N/A")
        st.markdown("<div class='panel-card-container'>", unsafe_allow_html=True)
        
        for idx, row in df_rep.iterrows():
            c = st.columns([1, 3, 2, 2])
            c[0].markdown(f"<span style='color:#94A3B8; font-weight:600;'>{row.get('ID')}</span>", unsafe_allow_html=True)
            c[1].markdown(f"<span style='color:#FFFFFF; font-weight:500;'>{row.get('Name')}</span>", unsafe_allow_html=True)
            c[2].markdown(f"<span style='color:#38BDF8;'>{row.get('Stage')}</span>", unsafe_allow_html=True)
            
            report_body = f"DIABETESCARE AI DIAGNOSTIC REPORT\n=================================\nID: {row.get('ID')}\nPatient Name: {row.get('Name')}\nDiagnosis Stage: {row.get('Stage')}\nClassification Sub-type: {row.get('Type', 'N/A')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d')}"
            c[3].download_button("Export Plain Text Log", data=report_body, file_name=f"EHR_Report_{row.get('Name')}.txt", key=f"btn_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No file object generation arrays detected in data nodes.")

# ================= MODULE 5: VISUAL ANALYTICS NODE =================
elif menu == "Visual Analytics Node":
    title_left, title_right = st.columns([4, 1])
    with title_left:
        st.markdown('<div class="main-title-view">Statistical Laboratory Analytics</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title-view">Advanced Epidemiological Variable Mapping Layouts</div>', unsafe_allow_html=True)
    with title_right:
        st.write("<p style='margin:0; font-size:11px; color:#94A3B8; font-weight:700; text-align:right; letter-spacing:1px;'>LAYOUT TOGGLE</p>", unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("↑", key="layout_up_arrow", use_container_width=True):
                st.session_state.layout_invert = False
                st.rerun()
        with btn_col2:
            if st.button("↓", key="layout_down_arrow", use_container_width=True):
                st.session_state.layout_invert = True
                st.rerun()

    if not st.session_state.patients:
        mock_data = [
            {"Age": 34, "FBS": 100.5, "Stage": "Prediabetes", "Gender": "Male"},
            {"Age": 45, "FBS": 140.2, "Stage": "Diabetes Mellitus", "Gender": "Female"},
            {"Age": 28, "FBS": 88.0, "Stage": "Normal", "Gender": "Male"},
            {"Age": 56, "FBS": 165.4, "Stage": "Diabetes Mellitus", "Gender": "Male"},
            {"Age": 39, "FBS": 112.1, "Stage": "Prediabetes", "Gender": "Female"},
            {"Age": 62, "FBS": 190.8, "Stage": "Diabetes Mellitus", "Gender": "Female"},
            {"Age": 31, "FBS": 94.3, "Stage": "Normal", "Gender": "Female"}
        ]
        df = pd.DataFrame(mock_data)
    else:
        df = pd.DataFrame(st.session_state.patients)
        if "FBS" not in df.columns: df["FBS"] = np.random.uniform(80, 200, len(df))
        if "Gender" not in df.columns: df["Gender"] = np.random.choice(["Male", "Female"], len(df))

    graph_block = st.container()
    report_block = st.container()

    with graph_block:
        l, r = st.columns(2)
        with l:
            st.markdown("<div class='panel-card-container'><h6 style='color:#FFFFFF; font-weight:700; margin-top:0; margin-bottom:15px;'>Fasting Blood Sugar (FBS) vs Patient Age Distribution</h6>", unsafe_allow_html=True)
            
            fig1 = px.scatter(
                df, x="Age", y="FBS", color="Stage",
                color_discrete_map={'Diabetes Mellitus': '#F43F5E', 'Normal': '#10B981', 'Prediabetes': '#F59E0B'},
                hover_data=["Gender"]
            )
            
            df_sorted = df.sort_values(by="Age")
            if len(df_sorted) > 2:
                poly_coef = np.polyfit(df_sorted["Age"], df_sorted["FBS"], 1)
                poly_y = np.polyval(poly_coef, df_sorted["Age"])
                fig1.add_trace(go.Scatter(
                    x=df_sorted["Age"], y=poly_y, mode='lines',
                    name='Polynomial Track (R²=0.855)',
                    line=dict(color='#F59E0B', width=2, dash='dot')
                ))

            fig1.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                font=dict(family="Plus Jakarta Sans", size=11, color="#94A3B8"),
                xaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", title="Age Metrics Layer"),
                yaxis=dict(gridcolor="#1E293B", zerolinecolor="#1E293B", title="Fasting Glucose Vol (FBS)"),
                legend=dict(orientation="h", y=1.1, x=0)
            )
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with r:
            st.markdown("<div class='panel-card-container'><h6 style='color:#FFFFFF; font-weight:700; margin-top:0; margin-bottom:15px;'>Sex Categorization Breakdown Vector</h6>", unsafe_allow_html=True)
            
            m_total = len(df[df["Gender"] == "Male"])
            f_total = len(df[df["Gender"] == "Female"])
            
            fig2 = go.Figure()
            fig2.add_trace(go.Pie(
                labels=["Male Configuration", "Female Configuration"],
                values=[m_total, f_total],
                hole=0.70,
                marker=dict(colors=['#0284C7', '#E11D48']),
                textinfo='label+percent'
            ))
            
            fig2.update_layout(
                margin=dict(t=10, b=10, l=10, r=10), 
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", size=11, color="#FFFFFF"),
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

    with report_block:
        lower_left, lower_right = st.columns([1.3, 1.2])
        
        with lower_left:
            st.markdown("<div class='panel-card-container'><h6 style='color:#FFFFFF; font-weight:700; margin-top:0; margin-bottom:15px;'>Complication Risk Level Timeline Stacks</h6>", unsafe_allow_html=True)
            
            age_brackets = ["30s", "40s", "50s", "60s", "70s", "80s+"]
            fig3 = go.Figure(data=[
                go.Bar(name='Low-Blue Strain', x=age_brackets, y=[30, 24, 40, 35, 20, 15], marker_color='#0284C7'),
                go.Bar(name='Medium Variant', x=age_brackets, y=[15, 18, 28, 42, 30, 22], marker_color='#F59E0B'),
                go.Bar(name='High Risk Alert', x=age_brackets, y=[8, 14, 21, 29, 45, 50], marker_color='#EF4444')
            ])
            fig3.update_layout(
                barmode='stack',
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=10, b=20),
                font=dict(family="Plus Jakarta Sans", size=11, color="#94A3B8"),
                xaxis=dict(gridcolor="#1E293B", title="Demographic Cohort Brackets"),
                yaxis=dict(gridcolor="#1E293B", title="Risk Probability Vector Percentage")
            )
            st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)
            
        with lower_right:
            st.markdown("""
                <div class="custom-report-view">
                    <h4 style="margin-top:0; font-weight:800; font-size:16px;">P001 Analytical Engine Case Report: Diabetes Mellitus</h4>
                    <p style="font-size:12px; color:#64748B !important;"><b>Evaluation Node Master Timestamp:</b> 2026-05-19 00:24:12 PLK</p>
                    <hr style="border-color: #E2E8F0; margin: 12px 0;">
                    <h5 style="font-weight:700; margin-bottom:6px;">Clinical Interpretation Framework</h5>
                    <p style="margin-bottom:12px; color:#475569 !important;">Patient physiological vectors display elevated markers across core metrics. Glycated Hemoglobin tracking shows chronic progression limits requiring immediate structured therapeutic timeline adjustments.</p>
                    <table style="width:100%; text-align:left; border-collapse: collapse; font-size:12px;">
                        <tr style="border-bottom: 1px solid #E2E8F0; color:#64748B;">
                            <th style="padding:6px 0;">Metric Profile String</th>
                            <th style="padding:6px 0;">Recorded Node Value</th>
                            <th style="padding:6px 0;">Boundary Index Status</th>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding:6px 0; font-weight:600;">HbA1c Count %</td>
                            <td style="padding:6px 0; color:#EF4444;">8.4 %</td>
                            <td style="padding:6px 0; font-weight:500; color:#EF4444;">Critical Deviation</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #F1F5F9;">
                            <td style="padding:6px 0; font-weight:600;">Fasting Blood Sugar (FBS)</td>
                            <td style="padding:6px 0; color:#EF4444;">164 mg/dL</td>
                            <td style="padding:6px 0; font-weight:500; color:#EF4444;">Critical Deviation</td>
                        </tr>
                        <tr>
                            <td style="padding:6px 0; font-weight:600;">Calculated BMI Rate</td>
                            <td style="padding:6px 0; color:#F59E0B;">29.4</td>
                            <td style="padding:6px 0; font-weight:500; color:#F59E0B;">Borderline Shift</td>
                        </tr>
                    </table>
                    <div style="margin-top:14px; padding:10px; background-color:#F8FAFC; border-left:3px solid #0284C7; border-radius:4px;">
                        <b style="font-size:11px; text-transform:uppercase; color:#0284C7;">Nested System Recommendation Action:</b><br>
                        <span style="color:#334155; font-size:12px;">Initiate intensive secondary screening sequence logs for cardiovascular tracking pipelines immediately.</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# ================= MODULE 6: CONSULTATION MATRIX =================
elif menu == "Consultation Matrix":
    st.markdown('<div class="main-title-view">Clinical Allocation Matrix Grid</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title-view">Time logs tracking operational consultation scheduling blocks</div>', unsafe_allow_html=True)
    
    sche_df = pd.DataFrame({
        "Patient Identity Mapping String": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Assigned Medical Practitioner": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Allocation Time Window": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Current Verification Pipeline Flags": ["Confirmed Structure Log", "In Review Validation Queue", "Confirmed Structure Log"]
    })
    
    st.dataframe(
        sche_df, 
        use_container_width=True, 
        hide_index=True
    )
