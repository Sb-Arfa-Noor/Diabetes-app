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

# ================= SESSION STATE FOR LOGIN =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= MODERN INTERACTIVE UI STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B !important;
    }
    section[data-testid="stSidebar"] * {
        color: #E2E8F0 !important;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Custom Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        border-left: 5px solid #0284C7;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #64748B;
        font-size: 13px;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #1E293B;
        margin-top: 5px;
    }
    
    /* Login Background Panel */
    .login-panel {
        background: white;
        padding: 35px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* Section Block Container */
    .custom-container {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.01);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    /* Clean Title Text */
    .main-title {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 25px;
    }
    
    /* Custom Navigation Buttons Style */
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0369A1;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ================= BEAUTIFUL SIMPLE LOGIN FORM =================
if not st.session_state.logged_in:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("""
            <div class='login-panel'>
                <h2 style='text-align: center; color: #0F172A; font-weight:700; margin-bottom: 5px;'>Clinical Portal Login</h2>
                <p style='text-align: center; color: #64748B; font-size:13px; margin-bottom: 25px;'>Please enter your credentials to access the system</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Form Container Inputs
        username = st.text_input("Username", value="admin", placeholder="Enter username...")
        password = st.text_input("Password", type="password", placeholder="Enter password...")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign In to Portal", use_container_width=True):
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.success("Login Successful!")
                st.rerun()
            else:
                st.error("Invalid Username or Password")
    st.stop()

# ================= SIDEBAR NAVIGATION =================
st.sidebar.markdown("""
<div style='padding: 10px 0px;'>
    <h2 style='color: white; font-weight: 700; margin-bottom: 0px;'>DiabetesCare AI</h2>
    <p style='color: #94A3B8; font-size: 12px; margin-top: 0px;'>Smart Diagnostics App</p>
</div>
<hr style='border-color: #334155; margin-top: 0px;'>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard", "Patients List", "New Diagnosis", "Export Reports", "Visual Analytics", "Appointments"]
)

st.sidebar.markdown("<br><br><hr style='border-color: #334155;'>", unsafe_allow_html=True)
if st.sidebar.button("Logout", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

# ================= MENU 1: DASHBOARD =================
if menu == "Dashboard":
    st.markdown('<div class="main-title">System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Overview of current registered clinical database records</div>', unsafe_allow_html=True)
    
    # Summary Metrics Calculate
    total = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])
    normal = total - positive - risk
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card" style="border-left-color: #0284C7;"><div class="metric-title">Total Cases</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card" style="border-left-color: #DC2626;"><div class="metric-title">Diabetes Patients</div><div class="metric-value">{positive}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card" style="border-left-color: #D97706;"><div class="metric-title">Prediabetes Risk</div><div class="metric-value">{risk}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card" style="border-left-color: #0D9488;"><div class="metric-title">Normal Status</div><div class="metric-value">{normal}</div></div>', unsafe_allow_html=True)
        
    st.write("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.6, 1])
    with col1:
        st.markdown("#### Recent Case Logs")
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients)
            df = df.fillna("N/A")
            st.dataframe(df[["ID", "Name", "Age", "Gender", "Stage"]], use_container_width=True, hide_index=True)
        else:
            st.info("No records available in database.")
            
    with col2:
        st.markdown("#### Patient Ratio Analysis")
        fig = go.Figure(data=[go.Pie(
            labels=["Diabetes", "Normal", "Prediabetes"],
            values=[positive, normal, risk],
            hole=.6,
            marker=dict(colors=['#DC2626', '#0D9488', '#D97706'])
        )])
        fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

# ================= MENU 2: NEW DIAGNOSIS (WIZARD FORM) =================
elif menu == "New Diagnosis":
    if "step" not in st.session_state:
        st.session_state.step = 1
    
    st.markdown('<div class="main-title">AI Diagnostic Engine</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">Multi-stage predictive analytics pipeline — <b>Currently on Step {st.session_state.step} of 3</b></div>', unsafe_allow_html=True)
    
    # STEP 1: Core Inputs
    if st.session_state.step == 1:
        st.markdown("<div class='custom-container'><h5>Step 1: Patient Details & Blood Report Values</h5><br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            name = st.text_input("Patient Full Name", placeholder="Enter name here...")
            age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.5)
        with r:
            fbs = st.number_input("Fasting Blood Sugar (FBS) [mg/dL]", min_value=50.0, max_value=400.0, value=100.0)
            hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.8)
            gtt = st.number_input("2-Hour Glucose Test (GTT) [mg/dL]", min_value=50.0, max_value=500.0, value=130.0)
            contact = st.text_input("Contact Number", placeholder="Enter phone number...")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if st.button("Next: Diagnostic Analysis"):
            if not name.strip():
                st.error("Please enter patient name to proceed.")
            else:
                # Core Calculations
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
                    # Direct Save for Normal/Prediabetes
                    p = st.session_state.patient_data.copy()
                    p["ID"] = f"P{len(st.session_state.patients)+1}"
                    p["Type"] = "N/A"
                    p["Complications"] = "None"
                    p["Date"] = datetime.now().strftime("%Y-%m-%d")
                    add_patient(p)
                    st.session_state.patients = get_patients()
                    st.session_state.step = 3
                st.rerun()

    # STEP 2: Advanced Clinical Profile (For Positive Cases Only)
    elif st.session_state.step == 2:
        st.markdown("<div class='custom-container'><h5>Step 2: Advanced Labs (Etiology & Type Prediction)</h5><br>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            pregnant = st.selectbox("Is Patient Pregnant?", [0, 1], help="0 = No, 1 = Yes")
            insulin = st.number_input("Total Insulin Intake Units (If any)", value=0.0)
            duration = st.number_input("Duration of Diabetes Symptoms (Years)", value=0.0)
            urea = st.number_input("Serum Urea level", value=25.0)
        with r:
            antigad = st.selectbox("Anti-GAD Antibody Test", [0, 1], help="0 = Negative, 1 = Positive")
            ica = st.selectbox("Islet Cell Antibody (ICA) Test", [0, 1], help="0 = Negative, 1 = Positive")
            creatinine = st.number_input("Serum Creatinine level", value=0.8)
            smoker = st.selectbox("Is Patient a Smoker?", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Back"):
                st.session_state.step = 1
                st.rerun()
        with c2:
            if st.button("Next: Complications Map"):
                # Append data and process ML type prediction
                st.session_state.patient_data.update({
                    "Pregnant": pregnant, "InsulinTotalUnits": insulin, "DMDuration": duration,
                    "Urea": urea, "AntiGad": antigad, "ICA": ica, "Creatinine": creatinine, "Smoker": smoker,
                    "HistoryOfGDM": 0, "Triglycerides": 150.0, "DMonSetAge": st.session_state.patient_data["Age"], "IA2A": 0
                })
                
                # Basic Rule/ML prediction fallback mapping
                p_data = st.session_state.patient_data
                if antigad == 1 or ica == 1: dtype = "Type 1 Diabetes"
                elif pregnant == 1: dtype = "Gestational Diabetes"
                else: dtype = "Type 2 Diabetes"
                
                st.session_state.patient_data["Type"] = dtype
                st.session_state.step = 3
                st.rerun()

    # STEP 3: Summary & Final Report Setup
    elif st.session_state.step == 3:
        st.markdown("<div class='custom-container'><h5>Step 3: Diagnostic Report Summary</h5>", unsafe_allow_html=True)
        p = st.session_state.patient_data
        
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 20px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <h4 style="color: #0284C7; margin-top: 0;">Diagnostic Output Summary</h4>
                <hr style="margin: 10px 0;">
                <p><b>Patient Name:</b> {p.get('Name')}</p>
                <p><b>Age / Gender:</b> {p.get('Age')} Years old | {p.get('Gender')}</p>
                <p><b>HbA1c Lab Level:</b> {p.get('HbA1c')}%</p>
                <p><b>Clinical Diagnosis State:</b> <span style="color: #DC2626; font-weight: 700;">{p.get('Stage')}</span></p>
                <p><b>Diabetes Sub-type Classification:</b> {p.get('Type', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        if p.get("Stage") == "Diabetes Mellitus" and "ID" not in p:
            if st.button("Save Entry & Print File Log"):
                p["ID"] = f"P{len(st.session_state.patients)+1}"
                p["Complications"] = "Retinopathy Risk Detected" if p.get("DMDuration", 0) > 5 else "No High Structural Risks"
                p["Date"] = datetime.now().strftime("%Y-%m-%d")
                add_patient(p)
                st.session_state.patients = get_patients()
                st.success("Record successfully saved to system memory.")
                
        if st.button("Start New Diagnosis Session"):
            st.session_state.step = 1
            st.session_state.patient_data = {}
            st.rerun()

# ================= MENU 3: PATIENTS LIST =================
elif menu == "Patients List":
    st.markdown('<div class="main-title">EHR Patients Database</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Full ledger registry query controls</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        df = df.fillna("N/A")
        
        search = st.text_input("🔍 Quick Search Patient by Name String", value="")
        if search:
            df = df[df["Name"].str.contains(search, case=False, na=False)]
            
        st.dataframe(df[["ID", "Name", "Age", "Gender", "Contact", "Stage", "Type"]], use_container_width=True, hide_index=True)
        
        st.write("<br>", unsafe_allow_html=True)
        st.markdown("##### Remove or Purge Records")
        del_id = st.text_input("Enter Patient ID to delete (e.g. P1)", placeholder="Enter ID...")
        if st.button("Delete Patient Record"):
            if del_id:
                delete_patient(del_id)
                st.session_state.patients = get_patients()
                st.success(f"Record {del_id} cleared successfully.")
                st.rerun()
    else:
        st.info("The system database holds no active records currently.")

# ================= MENU 4: EXPORT REPORTS =================
elif menu == "Export Reports":
    st.markdown('<div class="main-title">Export Case Summaries</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Download plain text official verification files</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
        for idx, row in pd.DataFrame(st.session_state.patients).iterrows():
            c = st.columns([1, 3, 2, 2])
            c[0].write(f"**{row.get('ID', 'P')}**")
            c[1].write(row.get("Name"))
            c[2].write(row.get("Stage"))
            
            report_body = f"DIABETESCARE AI DIAGNOSTIC REPORT\n=================================\nID: {row.get('ID')}\nPatient Name: {row.get('Name')}\nDiagnosis Stage: {row.get('Stage')}\nClassification Sub-type: {row.get('Type', 'N/A')}\nTimestamp: {datetime.now().strftime('%Y-%m-%d')}"
            c[3].download_button("Download TXT File", data=report_body, file_name=f"Report_{row.get('Name')}.txt", key=f"btn_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No records to export.")

# ================= MENU 5: VISUAL ANALYTICS =================
elif menu == "Visual Analytics":
    st.markdown('<div class="main-title">Analytical Lab Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Epidemiological charts and graphics</div>', unsafe_allow_html=True)
    
    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        
        l, r = st.columns(2)
        with l:
            st.write("###### Blood Sugar (FBS) vs Age Metrics Scatter")
            fig1 = px.scatter(df, x="Age", y="FBS", color="Stage", color_discrete_sequence=['#DC2626', '#0D9488', '#D97706'], template="simple_white")
            st.plotly_chart(fig1, use_container_width=True)
        with r:
            st.write("###### Gender Distribution Plot")
            m_count = len(df[df["Gender"] == "Male"])
            f_count = len(df[df["Gender"] == "Female"])
            fig2 = go.Figure(data=[go.Pie(labels=["Male", "Female"], values=[m_count, f_count], hole=0.5, marker=dict(colors=['#0284C7','#E11D48']))])
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Charts are unavailable until records populate arrays.")

# ================= MENU 6: APPOINTMENTS =================
elif menu == "Appointments":
    st.markdown('<div class="main-title">Scheduler & Booking Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Active time-slot matrix maps</div>', unsafe_allow_html=True)
    
    sche_df = pd.DataFrame({
        "Patient Identifier Key": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Assigned Medical Doctor": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Time Allotment": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Approval Queue State": ["Approved", "In Queue Process", "Approved"]
    })
    st.markdown("<div class='custom-container'>", unsafe_allow_html=True)
    st.dataframe(sche_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
