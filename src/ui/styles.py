"""Custom CSS styles for the Streamlit application."""

import streamlit as st


def apply_custom_css():
    """Apply professional academic conference CSS styling."""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container */
    .main {
        background: #f8f9fa !important;
        padding: 2rem 1rem;
    }

    /* Force light theme */
    .stApp {
        background: #f8f9fa !important;
    }

    /* Header */
    .main-header {
        text-align: center;
        color: #1a202c !important;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-header {
        text-align: center;
        color: #2d3748 !important;
        font-size: 1.1rem;
        margin-bottom: 3rem;
        font-weight: 500;
        animation: fadeInUp 0.8s ease-out;
        line-height: 1.6;
    }

    /* Card Styles */
    .stApp > div > div > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }

    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    .stButton>button:active {
        transform: translateY(-1px);
    }

    /* File Uploader */
    .stFileUploader {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed #cbd5e0;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: #667eea;
        background: #f7fafc;
    }

    /* Radio Buttons */
    .stRadio > label {
        font-weight: 600;
        color: #2d3748;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    .stRadio > div {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Checkbox */
    .stCheckbox {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    /* Sidebar - Modern Professional Style */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #2c3e50 100%) !important;
        padding: 0 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1rem 1rem 2rem 1rem !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] .stMarkdown p {
        color: #ffffff !important;
    }

    [data-testid="stSidebar"] h2 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700;
        font-size: 1.3rem;
    }

    /* Sidebar Radio Buttons - Modern Style with proper alignment */
    [data-testid="stSidebar"] .stRadio {
        margin: 0 0 1.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stRadio > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%) !important;
        padding: 1.25rem !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        margin: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
        padding: 0.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stRadio > label {
        color: #ffffff !important;
        font-weight: 700 !important;
        margin-bottom: 0 !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stRadio > div > label {
        margin-bottom: 0.75rem !important;
    }

    /* Sidebar Checkbox - Modern Style with proper alignment */
    [data-testid="stSidebar"] .stCheckbox {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%) !important;
        padding: 1.25rem !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        border: 2px solid rgba(255,255,255,0.3) !important;
        transition: all 0.3s ease;
        margin: 0 0 1.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stCheckbox:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
    }

    [data-testid="stSidebar"] .stCheckbox label {
        color: #1a202c !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Sidebar Info/Success/Warning Boxes - Maximum Readability */
    [data-testid="stSidebar"] .stAlert,
    [data-testid="stSidebar"] [data-baseweb="notification"],
    [data-testid="stSidebar"] .element-container .stAlert {
        background: #ffffff !important;
        border-radius: 15px !important;
        padding: 1.25rem !important;
        border: 2px solid #dee2e6 !important;
        border-left: 5px solid !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25) !important;
        margin: 0 0 1.5rem 0 !important;
    }

    [data-testid="stSidebar"] .stAlert *,
    [data-testid="stSidebar"] [data-baseweb="notification"] * {
        color: #000000 !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        font-weight: 500 !important;
        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
    }

    [data-testid="stSidebar"] .stAlert strong,
    [data-testid="stSidebar"] [data-baseweb="notification"] strong {
        color: #000000 !important;
        font-weight: 800 !important;
        font-size: 0.95rem !important;
    }

    /* Info Box - Blue Background */
    [data-testid="stSidebar"] [data-baseweb="notification"][kind="info"],
    [data-testid="stSidebar"] .stInfo {
        border-left-color: #0d6efd !important;
        background: #cfe2ff !important;
    }

    [data-testid="stSidebar"] [data-baseweb="notification"][kind="info"] *,
    [data-testid="stSidebar"] .stInfo * {
        color: #052c65 !important;
    }

    /* Success Box - Green Background */
    [data-testid="stSidebar"] [data-baseweb="notification"][kind="success"],
    [data-testid="stSidebar"] .stSuccess {
        border-left-color: #198754 !important;
        background: #d1e7dd !important;
    }

    [data-testid="stSidebar"] [data-baseweb="notification"][kind="success"] *,
    [data-testid="stSidebar"] .stSuccess * {
        color: #0a3622 !important;
    }

    /* Warning Box - Yellow Background */
    [data-testid="stSidebar"] [data-baseweb="notification"][kind="warning"],
    [data-testid="stSidebar"] .stWarning {
        border-left-color: #fd7e14 !important;
        background: #fff3cd !important;
    }

    [data-testid="stSidebar"] [data-baseweb="notification"][kind="warning"] *,
    [data-testid="stSidebar"] .stWarning * {
        color: #664d03 !important;
    }

    /* Success/Error Messages - High Contrast */
    .stSuccess {
        background: #d4edda !important;
        color: #155724 !important;
        border: 2px solid #28a745 !important;
        border-left: 5px solid #28a745 !important;
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 600;
        animation: slideInRight 0.5s ease-out;
    }

    .stSuccess p, .stSuccess strong, .stSuccess li {
        color: #155724 !important;
        font-weight: 500 !important;
    }

    .stError {
        background: #f8d7da !important;
        color: #721c24 !important;
        border: 2px solid #dc3545 !important;
        border-left: 5px solid #dc3545 !important;
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 600;
        animation: slideInRight 0.5s ease-out;
    }

    .stError p, .stError strong, .stError li {
        color: #721c24 !important;
        font-weight: 500 !important;
    }

    /* Info Box - High Contrast */
    .stInfo {
        background: #cfe2ff !important;
        color: #084298 !important;
        border: 2px solid #0d6efd !important;
        border-left: 5px solid #0d6efd !important;
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 500;
    }

    .stInfo p, .stInfo strong, .stInfo li {
        color: #084298 !important;
        font-weight: 500 !important;
    }

    /* Warning Box - High Contrast */
    .stWarning {
        background: #fff3cd !important;
        color: #664d03 !important;
        border: 2px solid #ffc107 !important;
        border-left: 5px solid #ffc107 !important;
        border-radius: 12px;
        padding: 1.2rem;
        font-weight: 500;
    }

    .stWarning p, .stWarning strong, .stWarning li {
        color: #664d03 !important;
        font-weight: 500 !important;
    }

    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e0, transparent);
    }

    /* Image Display */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }

    .stImage:hover {
        transform: scale(1.02);
    }

    /* Markdown Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    h3 {
        font-size: 1.5rem;
    }

    h4 {
        font-size: 1.25rem;
        color: #2d3748 !important;
    }

    /* Ensure all text is visible */
    p, li, span, div {
        color: #2d3748 !important;
    }

    /* Force readable text in ALL alert boxes */
    .stAlert, [data-testid*="stNotification"], [data-baseweb="notification"] {
        font-size: 0.95rem !important;
        line-height: 1.7 !important;
    }

    .stAlert *, [data-testid*="stNotification"] *, [data-baseweb="notification"] * {
        color: inherit !important;
    }

    .stAlert p, .stAlert li, .stAlert span,
    [data-testid*="stNotification"] p, [data-testid*="stNotification"] li, [data-testid*="stNotification"] span,
    [data-baseweb="notification"] p, [data-baseweb="notification"] li, [data-baseweb="notification"] span {
        margin: 0.5rem 0 !important;
        color: inherit !important;
    }

    .stAlert strong, [data-testid*="stNotification"] strong, [data-baseweb="notification"] strong {
        font-weight: 700 !important;
        color: inherit !important;
    }

    /* Tips Box */
    .stMarkdown ul {
        background: #f7fafc;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }

    /* Metric Cards */
    [data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)
