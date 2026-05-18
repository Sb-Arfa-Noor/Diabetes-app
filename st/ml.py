# ================= IMPORTS =================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
from database import create_table, get_patients, add_patient, delete_patient

# ================= LOAD ML MODELS =================
model1 = joblib.load("ml_models/step1.pkl")
model2 = joblib.load("ml_models/step2.pkl")
model3 = joblib.load("ml_models/step3.pkl")

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

create_table()

# ================= SESSION STATE MANAGEMENT =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "auth_page" not in st.session_state:
    st.session_state.auth_page = "signin"  # 'signin' or 'signup'

if "user_db" not in st.session_state:
    st.session_state.user_db = {"admin": "password"}  # Default presentation user

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

# ================= HIGH-END PREMIUM CSS STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Overrides */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    
    /* Auth Form Premium Card */
    .auth-card {
        background: #FFFFFF;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
        border: 1px solid #E2E8F0;
        margin-top: 30px;
    }
    .auth-title {
        font-size: 28px;
        font-weight: 700;
        color: #0F172A;
        text-align: center;
        letter-spacing: -0.5px;
    }
    .auth-subtitle {
        font-size: 14px;
        color: #64748B;
        text-align: center;
        margin-bottom: 30px;
    }
    
    /* Sidebar Elegant Look */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
        border-right: 1px solid #1E293B;
    }
    section[data-testid="stSidebar"] * {
        color: #94A3B8 !important;
    }
    
    /* Clean Navigation Choices */
    div[data-testid="stRadio"] label {
        padding: 12px 16px !important;
        border-radius: 8px !important;
        margin-bottom: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:hover {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    
    /* Structural Section Layouts */
    .app-title {
        font-size: 32px;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
    }
    .app-subtitle {
        font-size: 15px;
        color: #64748B;
        margin-bottom: 30px;
    }
    
    /* KPI Dashboard Cards */
    .kpi-box {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.01);
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: #0F172A;
        margin-top: 4px;
    }
    
    /* Custom Decent Form Containers */
    .form-container {
        background: #FFFFFF;
        padding: 30px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 25px;
    }
    .form-header {
        font-size: 18px;
        font-weight: 600;
        color: #1E293B;
        padding-bottom: 12px;
        border-bottom: 1px solid #F1F5F9;
        margin-bottom: 20px;
    }
    
    /* Interactive Wizard Progress Steps Bar */
    .step-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 35px;
    }
    .step-item {
        text-align: center;
        flex: 1;
    }
    .step-dot {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 8px auto;
        font-weight: 600;
        font-size: 14px;
    }
    .step-active { background: #0F766E; color: #FFFFFF; }
    .step-pending { background: #E2E8F0; color: #94A3B8; }
    .step-complete { background: #115E59; color: #FFFFFF; }
    .step-text { font-size: 13px; font-weight: 500; color: #334155; }
    .step-line { height: 2px; background: #E2E8F0; flex: 1; margin-top: -28px; mx: 10px; }

    /* Override Streamlit Primary Buttons */
    .stButton>button {
        background-color: #0F766E;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #115E59;
    }
</style>
""", unsafe_allow_html=True)

# ================= AUTHENTICATION GATE (SIGN IN / SIGN UP) =================
if not st.session_state.logged_in:
    # Sidebar ko completely hide kar dete hain jab tak user login nahi hota
    st.markdown("""<style>section[data-testid="stSidebar"] {display: none !important;}</style>""", unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.1, 1.2, 1.1])
    
    with col2:
        if st.session_state.auth_page == "signin":
            st.markdown("""
                <div class="auth-card">
                    <div class="auth-title">Clinical Portal Access</div>
                    <div class="auth-subtitle">Provide your credentials to enter the system dashboard</div>
                </div>
            """, unsafe_allow_html=True)
            
            user_input = st.text_input("Username", key="si_user", placeholder="Enter username")
            pass_input = st.text_input("Password", type="password", key="si_pass", placeholder="Enter password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In", use_container_width=True):
                if user_input in st.session_state.user_db and st.session_state.user_db[user_input] == pass_input:
                    st.session_state.logged_in = True
                    st.rerun()
                else:
                    st.error("Authentication failed. Invalid username or password.")
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:20px;'>New practitioner?</p>", unsafe_allow_html=True)
            if st.button("Create an Account", use_container_width=True):
                st.session_state.auth_page = "signup"
                st.rerun()
                
        elif st.session_state.auth_page == "signup":
            st.markdown("""
                <div class="auth-card">
                    <div class="auth-title">Practitioner Registration</div>
                    <div class="auth-subtitle">Create new account credentials for secure workspace logging</div>
                </div>
            """, unsafe_allow_html=True)
            
            new_user = st.text_input("Choose Username", key="su_user", placeholder="e.g. dr_ahmed")
            new_pass = st.text_input("Choose Password", type="password", key="su_pass", placeholder="Minimum 6 characters")
            conf_pass = st.text_input("Confirm Password", type="password", key="su_conf", placeholder="Repeat chosen password")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register Account", use_container_width=True):
                if not new_user.strip() or not new_pass.strip():
                    st.error("Please fill in all requested registration parameters.")
                elif new_user in st.session_state.user_db:
                    st.error("This username key is already taken.")
                elif new_pass != conf_pass:
                    st.error("Password configurations do not match.")
                else:
                    st.session_state.user_db[new_user] = new_pass
                    st.success("Registration complete! Please proceed to login.")
                    st.session_state.auth_page = "signin"
                    st.rerun()
            
            st.markdown("<p style='text-align:center; font-size:13px; color:#64748B; margin-top:20px;'>Already registered?</p>", unsafe_allow_html=True)
            if st.button("Back to Sign In", use_container_width=True):
                st.session_state.auth_page = "signin"
                st.rerun()
    st.stop()

# ================= SIDEBAR NAVIGATION ARCHITECTURE =================
st.sidebar.markdown("""
<div style="padding: 20px 8px 10px 8px;">
    <div style="font-size:22px; font-weight:700; color:#F8FAFC; letter-spacing:-0.5px;">DiabetesCare AI</div>
    <div style="color:#64748B; font-size:12px; font-weight:500; margin-top:2px;">Clinical Management Unit</div>
</div>
<hr style="border-color: #1E293B; margin-top:10px; margin-bottom:15px;">
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation System",
    ["Dashboard", "Patients Registry", "Diagnostic Pipeline", "Case Reports", "Analytics Panel", "Scheduler Matrix"]
)

st.sidebar.markdown("<br><br><hr style='border-color: #1E293B;'>", unsafe_allow_html=True)
if st.sidebar.button("Log Out System", use_container_width=True):
    st.session_state.logged_in = False
    st.session_state.auth_page = "signin"
    st.rerun()

# ================= MENU 1: DASHBOARD =================
if menu == "Dashboard":
    left, right = st.columns([6, 1.5])
    with left:
        st.markdown('<div class="app-title">Dashboard Overview</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-subtitle">Real-time status tracking and clinical registry monitoring</div>', unsafe_allow_html=True)
    with right:
        current_date = datetime.now().strftime("%d %b %Y")
        st.markdown(f'<div class="kpi-box" style="padding:14px; text-align:center; font-weight:600; color:#475569; background:#FFFFFF;">{current_date}</div>', unsafe_allow_html=True)

    # Metrics Calculation
    total = len(st.session_state.patients)
    positive = len([p for p in st.session_state.patients if p.get("Stage") == "Diabetes Mellitus"])
    risk = len([p for p in st.session_state.patients if p.get("Stage") == "Prediabetes"])
    normal = total - positive - risk

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="kpi-box" style="border-top:4px solid #0F766E;"><div class="kpi-label">Total Records</div><div class="kpi-value">{total}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="kpi-box" style="border-top:4px solid #DC2626;"><div class="kpi-label">Confirmed Positive</div><div class="kpi-value">{positive}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="kpi-box" style="border-top:4px solid #D97706;"><div class="kpi-label">Prediabetes At Risk</div><div class="kpi-value">{risk}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="kpi-box" style="border-top:4px solid #0D9488;"><div class="kpi-label">Normal Status</div><div class="kpi-value">{normal}</div></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.7, 1])
    with col1:
        st.markdown("#### Recent Electronic Health Records")
        if st.session_state.patients:
            df = pd.DataFrame(st.session_state.patients).fillna("N/A")
            st.dataframe(df[["ID", "Name", "Age", "Gender", "Stage"]], use_container_width=True, hide_index=True)
        else:
            st.info("No current clinical cases recorded.")
    with col2:
        st.markdown("#### Patient Category Analysis")
        fig = go.Figure(data=[go.Pie(labels=["Positive", "Normal", "Prediabetes"], values=[positive, normal, risk], hole=.68, marker=dict(colors=['#DC2626', '#0D9488', '#D97706']))])
        fig.update_layout(height=260, paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ================= MENU 2: DIAGNOSTIC PIPELINE (WIZARD FORM) =================
elif menu == "Diagnostic Pipeline":
    if "step" not in st.session_state: st.session_state.step = 1
    step = st.session_state.step

    st.markdown('<div class="app-title">AI Multi-Stage Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Machine learning multi-phase patient screening model</div>', unsafe_allow_html=True)

    # Clean Interactive Step Bar
    s1, l1, s2, l2, s3 = st.columns([1, 0.6, 1, 0.6, 1])
    with s1:
        d_cls = "step-complete" if step > 1 else "step-active"
        icon = "✓" if step > 1 else "1"
        st.markdown(f'<div class="step-item"><div class="step-dot {d_cls}">{icon}</div><div class="step-text">Primary Screening</div></div>', unsafe_allow_html=True)
    with l1: st.markdown('<div class="step-line"></div>', unsafe_allow_html=True)
    with s2:
        d_cls = "step-complete" if step > 2 else ("step-active" if step == 2 else "step-pending")
        icon = "✓" if step > 2 else "2"
        st.markdown(f'<div class="step-item"><div class="step-dot {d_cls}">{icon}</div><div class="step-text">Etiological Isolation</div></div>', unsafe_allow_html=True)
    with l2: st.markdown('<div class="step-line"></div>', unsafe_allow_html=True)
    with s3:
        d_cls = "step-active" if step == 3 else "step-pending"
        st.markdown(f'<div class="step-item"><div class="step-dot {d_cls}">3</div><div class="step-text">Complications Matrix</div></div>', unsafe_allow_html=True)

    st.write("<br>", unsafe_allow_html=True)

    # STEP 1
    if step == 1:
        st.markdown("<div class='form-container'><div class='form-header'>Patient Demographics & Glucose Blood Diagnostics</div>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            name = st.text_input("Patient Full Name", placeholder="Enter full name")
            age = st.number_input("Patient Age", min_value=1, max_value=120, value=35)
            gender = st.selectbox("Biological Sex Classification", ["Male", "Female"])
            bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, format="%.2f", value=23.4)
        with r:
            fbs = st.number_input("Fasting Blood Sugar (FBS) [mg/dL]", min_value=40.0, format="%.2f", value=100.0)
            hba1c = st.number_input("HbA1c Count Percentage (%)", min_value=3.0, format="%.2f", value=5.5)
            gtt = st.number_input("2-Hour Glucose Tolerance Test (GTT) [mg/dL]", min_value=40.0, format="%.2f", value=130.0)
            contact = st.text_input("Contact Number String", placeholder="e.g. 03001234567")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Analyze and Proceed"):
            if not name.strip(): st.error("Please input a valid identity name sequence.")
            else:
                stage = "Normal"
                if hba1c >= 6.5 or fbs >= 126 or gtt >= 200: stage = "Diabetes Mellitus"
                elif 5.7 <= hba1c < 6.5 or 100 <= fbs < 126 or 140 <= gtt < 200: stage = "Prediabetes"
                
                st.session_state.patient_data = {
                    "Name": name, "Age": age, "Gender": gender, "BMI": bmi, 
                    "FBS": fbs, "HbA1c": hba1c, "GTT2Hr": gtt, "Contact": contact, "Stage": stage
                }
                
                if stage == "Diabetes Mellitus": st.session_state.step = 2
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
    elif step == 2:
        st.markdown("<div class='form-container'><div class='form-header'>Advanced Autoantibody Panels & Pancreatic Indicators</div>", unsafe_allow_html=True)
        l, r = st.columns(2)
        with l:
            pregnant = st.selectbox("Gestational Pregnancy Status Vector", [0, 1])
            insulin = st.number_input("Exogenous Insulin Load Volume Units", value=0.0)
            duration = st.number_input("Known Diabetes Historical Duration (Years)", value=0.0)
            urea = st.number_input("Serum Blood Urea Clearance Level", value=24.0)
        with r:
            antigad = st.selectbox("Anti-GAD Autoantibody Screening Test Index", [0, 1])
            ica = st.selectbox("Islet Cell Cytoplasmic Antibody Assay Index", [0, 1])
            creatinine = st.number_input("Serum Creatinine Clearance Index", value=0.7)
            smoker = st.selectbox("Nicotine Tobacco Intake Factor Mapping", [0, 1])
        st.markdown("</div>", unsafe_allow_html=True)

        b, n = st.columns(2)
        if b.button("Previous Step"):
            st.session_state.step = 1
            st.rerun()
        if n.button("Process Etiology Type"):
            dtype = "Type 2 Diabetes"
            if antigad == 1 or ica == 1: dtype = "Type 1 Diabetes"
            elif pregnant == 1: dtype = "Gestational Diabetes"
            
            st.session_state.patient_data.update({
                "Pregnant": pregnant, "InsulinTotalUnits": insulin, "DMDuration": duration, "Urea": urea,
                "AntiGad": antigad, "ICA": ica, "Creatinine": creatinine, "Smoker": smoker,
                "HistoryOfGDM": 0, "Triglycerides": 140.0, "DMonSetAge": st.session_state.patient_data["Age"], "IA2A": 0
            })
            st.session_state.patient_data["Type"] = dtype
            st.session_state.step = 3
            st.rerun()

    # STEP 3
    elif step == 3:
        st.markdown("<div class='form-container'><div class='form-header'>Diagnostic Model Output Report Summary</div>", unsafe_allow_html=True)
        p = st.session_state.patient_data
        
        st.markdown(f"""
            <div style="background-color: #F8FAFC; padding: 24px; border-radius: 8px; border: 1px solid #E2E8F0;">
                <h5 style="color: #0F766E; margin-top: 0; font-weight:600; font-size:16px;">Pipeline Evaluation Success</h5>
                <hr style="margin: 12px 0; border-color:#E2E8F0;">
                <p style="margin:6px 0; font-size:14px;"><b>Patient Name Sequence:</b> {p.get('Name')}</p>
                <p style="margin:6px 0; font-size:14px;"><b>Demographic Context:</b> {p.get('Age')} Years | {p.get('Gender')}</p>
                <p style="margin:6px 0; font-size:14px;"><b>Primary Mapping Result:</b> <span style="color: #DC2626; font-weight:600;">{p.get('Stage')}</span></p>
                <p style="margin:6px 0; font-size:14px;"><b>Isolated Etiological Variant:</b> {p.get('Type', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if p.get("Stage") == "Diabetes Mellitus" and "ID" not in p:
            if st.button("Commit Case Record to Memory Node", use_container_width=True):
                p["ID"] = f"P{len(st.session_state.patients)+1}"
                p["Complications"] = "Retinopathy Strain Risk Evaluated" if p.get("DMDuration", 0) > 4 else "Baseline Systemic Matrix Clear"
                p["Date"] = datetime.now().strftime("%Y-%m-%d")
                add_patient(p)
                st.session_state.patients = get_patients()
                st.success("Case successfully integrated within static health logs.")

        if st.button("Reset Matrix Diagnostic Session"):
            st.session_state.step = 1
            st.session_state.patient_data = {}
            st.rerun()

# ================= MENU 3: PATIENTS REGISTRY =================
elif menu == "Patients Registry":
    st.markdown('<div class="app-title">Active Health Registry Records</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Query and drop configuration interfaces for historical patient records</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients).fillna("N/A")
        
        search = st.text_input("Filter Registry Records by Name Token")
        if search: df = df[df["Name"].str.contains(search, case=False, na=False)]
        
        st.dataframe(df[["ID", "Name", "Age", "Gender", "Contact", "Stage", "Type"]], use_container_width=True, hide_index=True)
        
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown("##### Administrative Operations Node")
        target_id = st.text_input("Target Unique ID to Purge (e.g., P1)")
        if st.button("Purge Entry Frame"):
            if target_id:
                delete_patient(target_id)
                st.session_state.patients = get_patients()
                st.success(f"Record {target_id} safely disconnected from structural storage matrices.")
                st.rerun()
    else:
        st.info("No compiled metrics logged inside current node cluster rows.")

# ================= MENU 4: CASE REPORTS =================
elif menu == "Case Reports":
    st.markdown('<div class="app-title">Documentation Export Nodes</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Download printable summary telemetry files</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        st.markdown("<div class='form-container'>", unsafe_allow_html=True)
        for index, row in pd.DataFrame(st.session_state.patients).iterrows():
            col_id, col_name, col_stage, col_action = st.columns([1, 3, 2, 2])
            col_id.write(f"**{row.get('ID', 'N/A')}**")
            col_name.write(row.get("Name"))
            col_stage.write(row.get("Stage"))
            
            raw_text = f"DIABETESCARE AI PLATFORM VERIFICATION LOG\n=========================================\nRecord Identifier Token: {row.get('ID')}\nFull Client Profile: {row.get('Name')}\nCalculated Condition Map: {row.get('Stage')}\nInferred Strain Variant: {row.get('Type', 'N/A')}\n"
            col_action.download_button("Download Text Log", data=raw_text, file_name=f"EHR_Report_{row.get('Name')}.txt", key=f"d_btn_{index}")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No documents are available to transfer to text file objects.")

# ================= MENU 5: ANALYTICS PANEL =================
elif menu == "Analytics Panel":
    st.markdown('<div class="app-title">Statistical Laboratory Dispersions</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Epidemiological variable mapping layouts</div>', unsafe_allow_html=True)

    if st.session_state.patients:
        df = pd.DataFrame(st.session_state.patients)
        
        c_left, c_right = st.columns(2)
        with c_left:
            st.write("###### Blood Sugar Level Scatter Matrix")
            fig1 = px.scatter(df, x="Age", y="FBS", color="Stage", color_discrete_sequence=['#DC2626', '#0D9488', '#D97706'], template="simple_white")
            st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        with c_right:
            st.write("###### Sex Category Proportions Profile")
            m_size = len(df[df["Gender"] == "Male"])
            f_size = len(df[df["Gender"] == "Female"])
            fig2 = go.Figure(data=[go.Pie(labels=["Male Group", "Female Group"], values=[m_size, f_size], hole=0.6, marker=dict(colors=['#0F766E','#BE123C']))])
            fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10))
            st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("Laboratory graphs are frozen until system values populate memory arrays.")

# ================= MENU 6: SCHEDULER MATRIX =================
elif menu == "Scheduler Matrix":
    st.markdown('<div class="app-title">Clinical Allocation Scheduling</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">Time tracking and operational appointment charts</div>', unsafe_allow_html=True)

    sched_df = pd.DataFrame({
        "Patient Identification Index": ["John Doe", "Sarah Khan", "Ali Raza"],
        "Assigned Healthcare Officer": ["Dr. Ahmed Khan", "Dr. Ahmed Khan", "Dr. Ahmed Khan"],
        "Target Allocation Slot": ["18 May - 10:00 AM", "19 May - 11:30 AM", "20 May - 02:00 PM"],
        "Status State Flag": ["Confirmed", "In Review Queue", "Confirmed"]
    })
    st.markdown("<div class='form-container'>", unsafe_allow_html=True)
    st.dataframe(sched_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
