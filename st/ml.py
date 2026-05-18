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
        "admin": "password"
    }

if "patients" not in st.session_state:
    st.session_state.patients = get_patients()

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = False

# ================= PREMIUM EXECUTIVE ULTRA CLEAN CSS STYLING =================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght=300;400;500;600;700;800&display=swap');
    
    /* Core Application Framework Theme */
    .stApp {
        background-color: #0B0F19;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, p, label, span, div {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Hide Streamlit's Native Built-in Sidebar Buttons completely */
    button[data-testid="sidebar-toggle-button"] {
        display: none !important;
    }

    /* CRITICAL FIX: Eliminate keyboard hints, shortcut strings and raw labels under buttons completely */
    span[data-testid="stWidgetHint"], 
    .stButton data-shortcut, 
    button p span,
    button[data-testid="baseButton-secondary"] span,
    div[data-testid="stWidgetLabel"] + div p,
    button[data-testid="baseButton-secondary"] p {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0px !important;
        height: 0px !important;
    }
    
    /* Main Streamlit Core Overrides */
    div[data-testid="stWidgetLabel"] p {
        color: #94A3B8 !important;
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
    
    /* ABSOLUTE REMOVAL OF SCROLLBARS FROM THE SIDE MENU CONTAINER */
    section[data-testid="stSidebar"] {
        background-color: #090D16 !important;
        border-right: 1px solid #1E293B !important;
        padding: 10px 14px !important;
        overflow: hidden !important;
    }
    
    section[data-testid="stSidebar"] div, 
    section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] .stBlock,
    section[data-testid="stSidebar"] [role="radiogroup"] {
        overflow: hidden !important;
        overflow-x: hidden !important;
        overflow-y: hidden !important;
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
        border-color: #38BDF8 !important;
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
        background: #111C44;
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
        color: #38BDF8 !important;
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
        background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%) !important;
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
    
    /* EXACT FIX: Absolute Position Placement For Custom Menu Toggle Arrows */
    div.fixed-collapse-container {
        position: fixed !important;
        top: 10px !important;
        left: 250px !important;
        z-index: 999999 !important;
    }
    div.fixed-expand-container {
        position: fixed !important;
        top: 10px !important;
        left: 15px !important;
        z-index: 999999 !important;
    }
    
    /* Raw Style overrides for custom arrow icons without text trace */
    .fixed-collapse-container button, .fixed-expand-container button {
        background: #111827 !important;
        border: 1px solid #1E293B !important;
        color: #38BDF8 !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 0px !important;
        margin: 0px !important;
        border-radius: 6px !important;
        width: 32px !important;
        min-width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Sidebar Lower Spacing Grid Configuration to clear absolute boundary lines */
    .sidebar-bottom-panel-padded {
        padding: 10px 12px 15px 12px !important;
        margin-top: 25px;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid #1E293B !important;
        border-radius: 8px !important;
        overflow: hidden;
    }
    
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

# ================= SIDEBAR RENDER ARCHITECTURE =================
if st.session_state.sidebar_collapsed:
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                width: 0px !important;
                min-width: 0px !important;
                padding: 0px !important;
                margin: 0px !important;
                border: none !important;
                visibility: hidden !important;
            }
        </style>
    """, unsafe_allow_html=True)
else:
    # PLACING COLLAPSE ARROW ON EXACT THE POSITION ENCLOSED BY RED MARKS
    st.markdown('<div class="fixed-collapse-container">', unsafe_allow_html=True)
    if st.button("‹", key="trigger_sidebar_collapse"):
        st.session_state.sidebar_collapsed = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown("""
    <div style='padding-top: 5px; margin-bottom: 10px;'>
        <div style='color: #FFFFFF; font-weight: 800; font-size:22px; letter-spacing:-0.75px;'>DiabetesCare AI</div>
        <div style='color: #38BDF8; font-size: 10px; margin-top: 2px; font-weight:700; text-transform: uppercase; letter-spacing:1px;'>Clinical Neural Center</div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<hr style='border-color: #1E293B; margin-top: 5px; margin-bottom:15px;'>", unsafe_allow_html=True)

    menu = st.sidebar.radio(
        "NAVIGATION NODE",
        ["Dashboard Overview", "Patients Matrix Registry", "Diagnostic Pipeline", "Data Report Center", "Visual Analytics Node", "Consultation Matrix"],
        label_visibility="collapsed"
    )

    # Lower Container Space With Protected Distance From Corner Lines
    st.sidebar.markdown("<div class='sidebar-bottom-panel-padded'>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border-color: #1E293B; margin-bottom: 16px;'>", unsafe_allow_html=True)
    if st.sidebar.button("Terminate Session Workspace", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.auth_screen = "login"
        st.rerun()
    st.sidebar.markdown("</div>
