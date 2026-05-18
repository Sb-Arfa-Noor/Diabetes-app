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
