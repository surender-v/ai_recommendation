"""
app.py  –  Student Risk Prediction & Personalised Recommendation Dashboard
Integrated with Random Forest ML Model & Groq LLM (Personalised AI Academic Mentor)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from groq import Groq

# ── Groq API Configuration ───────────────────────────────────────────────────
DEFAULT_GROQ_API_KEY = "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"
GROQ_MODEL = "qwen/qwen3.8-27b"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduRisk AI – Student Risk Predictor & LLM Mentor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Global font & smooth rendering */
    html, body, [class*="css"] { font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stSlider > div { color: #94a3b8 !important; }

    /* Main background */
    .main { background: #f8fafc; }

    /* Cards */
    .card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        box-shadow: 0 2px 4px rgba(15,23,42,0.05), 0 8px 24px rgba(15,23,42,0.04);
        margin-bottom: 18px;
        border: 1px solid #f1f5f9;
    }
    .card-title {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
    }

    /* Risk badges */
    .badge-high {
        background: #fee2e2; color: #991b1b; border: 1.5px solid #f87171;
        padding: 8px 22px; border-radius: 999px; font-weight: 800; font-size: 20px;
        display: inline-block; box-shadow: 0 2px 8px rgba(239,68,68,0.2);
    }
    .badge-medium {
        background: #fef3c7; color: #92400e; border: 1.5px solid #fcd34d;
        padding: 8px 22px; border-radius: 999px; font-weight: 800; font-size: 20px;
        display: inline-block; box-shadow: 0 2px 8px rgba(245,158,11,0.2);
    }
    .badge-low {
        background: #dcfce7; color: #166534; border: 1.5px solid #86efac;
        padding: 8px 22px; border-radius: 999px; font-weight: 800; font-size: 20px;
        display: inline-block; box-shadow: 0 2px 8px rgba(34,197,94,0.2);
    }

    /* Recommendation cards - High Contrast & Dark Cohesive Theme */
    .rec-card {
        border-left: 6px solid #6366f1 !important;
        background: #1e293b !important;
        padding: 16px 20px !important;
        border-radius: 0 12px 12px 0 !important;
        margin-bottom: 14px !important;
        font-size: 15px !important;
        color: #f8fafc !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18) !important;
        border: 1px solid #334155 !important;
        border-left-width: 6px !important;
    }
    .rec-card * {
        color: #f8fafc !important;
    }
    .rec-card b, .rec-card strong {
        color: #60a5fa !important;
        font-weight: 700 !important;
    }
    .rec-card-financial {
        border-left-color: #f59e0b !important;
        background: #1c1917 !important;
        border-color: #44403c !important;
    }
    .rec-card-financial b, .rec-card-financial strong {
        color: #fbbf24 !important;
    }
    .rec-card-academic {
        border-left-color: #3b82f6 !important;
        background: #0f172a !important;
        border-color: #1e293b !important;
    }
    .rec-card-academic b, .rec-card-academic strong {
        color: #93c5fd !important;
    }
    .rec-card-behavioral {
        border-left-color: #10b981 !important;
        background: #064e3b !important;
        border-color: #065f46 !important;
    }
    .rec-card-behavioral b, .rec-card-behavioral strong {
        color: #6ee7b7 !important;
    }

    /* LLM Recommendation Box - High Contrast, Deep Slate with Glowing Accents */
    .llm-box {
        background: #090d16 !important;
        border: 2px solid #6366f1 !important;
        border-radius: 16px !important;
        padding: 28px 32px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.35), 0 0 20px rgba(99,102,241,0.15) !important;
        line-height: 1.75 !important;
        color: #f1f5f9 !important;
    }
    .llm-box * {
        color: #f1f5f9 !important;
    }
    .llm-box h1, .llm-box h2, .llm-box h3, .llm-box h4 {
        margin-top: 22px !important;
        margin-bottom: 12px !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }
    .llm-box h1 {
        font-size: 24px !important;
        color: #a5b4fc !important;
        border-bottom: 1.5px solid #334155 !important;
        padding-bottom: 10px !important;
    }
    .llm-box h2 {
        font-size: 20px !important;
        color: #38bdf8 !important;
        border-bottom: 1px solid #1e293b !important;
        padding-bottom: 6px !important;
    }
    .llm-box h3 {
        font-size: 17px !important;
        color: #818cf8 !important;
    }
    .llm-box p {
        color: #e2e8f0 !important;
        font-size: 15px !important;
        margin-bottom: 14px !important;
        line-height: 1.7 !important;
    }
    .llm-box li {
        color: #e2e8f0 !important;
        font-size: 15px !important;
        margin-bottom: 8px !important;
        line-height: 1.6 !important;
    }
    .llm-box strong, .llm-box b {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .llm-box table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin: 18px 0 !important;
        background: #131b2e !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #334155 !important;
    }
    .llm-box th {
        background: #1e293b !important;
        color: #38bdf8 !important;
        padding: 12px 16px !important;
        font-weight: 700 !important;
        text-align: left !important;
        border-bottom: 2px solid #475569 !important;
        font-size: 14px !important;
    }
    .llm-box td {
        padding: 11px 16px !important;
        color: #f1f5f9 !important;
        border-bottom: 1px solid #1e293b !important;
        font-size: 14px !important;
    }
    .llm-box tr:nth-child(even) {
        background: #162035 !important;
    }
    .llm-box code {
        background: #1e293b !important;
        color: #38bdf8 !important;
        padding: 3px 7px !important;
        border-radius: 5px !important;
        font-size: 13.5px !important;
    }
    .llm-box hr {
        border: none !important;
        border-top: 1px solid #334155 !important;
        margin: 24px 0 !important;
    }

    /* Primary button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        width: 100% !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 16px rgba(99,102,241,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Load models & encoders ────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    target_dir = models_dir if os.path.isdir(models_dir) else base_dir

    model    = joblib.load(os.path.join(target_dir, 'risk_model.pkl'))
    encoder  = joblib.load(os.path.join(target_dir, 'label_encoder.pkl'))
    features = joblib.load(os.path.join(target_dir, 'feature_columns.pkl'))
    encoders = joblib.load(os.path.join(target_dir, 'encoders.pkl'))
    return model, encoder, features, encoders

model, label_enc, FEATURES, cat_encoders = load_artifacts()


# ── Initialize Groq Client ────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client(api_key: str):
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        return None


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg,#0f172a 0%, #312e81 100%);
            border-radius:16px; padding:28px 36px; margin-bottom:24px;
            box-shadow: 0 10px 25px -5px rgba(15,23,42,0.15);'>
    <div style='display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:16px;'>
        <div>
            <h1 style='color:white; margin:0; font-size:30px; font-weight:800; letter-spacing:-0.02em;'>
                🎓 EduRisk AI – Student Risk Predictor &amp; LLM Mentor
            </h1>
            <p style='color:#c7d2fe; margin:8px 0 0 0; font-size:15px;'>
                Machine Learning Early Warning System coupled with Groq-powered Personalised Learning Guidance
            </p>
        </div>
        <div style='background:rgba(255,255,255,0.1); border:1px solid rgba(255,255,255,0.2);
                    padding:8px 16px; border-radius:999px; color:#e0e7ff; font-size:13px; font-weight:600;'>
            ⚡ Powered by Random Forest + Groq (Qwen 27B)
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar – Student Profile Input ───────────────────────────────────────────
st.sidebar.markdown("## 📝 Student Profile")
st.sidebar.markdown("Enter student academic and personal indicators:")

with st.sidebar:
    # Optional Groq settings
    with st.expander("🔑 AI Settings (Groq)", expanded=False):
        api_key = st.text_input(
            "Groq API Key",
            value=DEFAULT_GROQ_API_KEY,
            type="password",
            help="Pre-configured with your Groq API key"
        )
        selected_model = st.selectbox(
            "LLM Model",
            ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
            index=0
        )

    st.markdown("### 👤 Personal Details")
    age        = st.slider("Age", 17, 30, 20)
    gender     = st.selectbox("Gender", ["Male", "Female"])
    department = st.selectbox("Department", [
        "Computer Science", "Electronics", "Mechanical", "Civil", "Information Technology"
    ])
    residence  = st.selectbox("Residence", ["Hostel", "Day Scholar"])
    guardian   = st.selectbox("Guardian", ["Father", "Mother", "Guardian"])
    family_inc = st.selectbox("Family Income Band", ["Low", "Medium", "High"])
    internet   = st.selectbox("Internet Access at Home", ["Yes", "No"])

    st.markdown("---")
    st.markdown("### 📊 Academic Marks")
    cat1        = st.slider("CAT 1 Marks (out of 50)", 0.0, 50.0, 24.0, step=0.5)
    cat2        = st.slider("CAT 2 Marks (out of 50)", 0.0, 50.0, 26.0, step=0.5)
    model_marks = st.slider("Model Exam Marks (out of 100)", 0.0, 100.0, 50.0, step=0.5)

    # Automatically compute weighted average on 0-100 scale: (CAT1*2 + CAT2*2 + Model)/3
    avg = round(((cat1 * 2) + (cat2 * 2) + model_marks) / 3, 2)
    st.metric("Computed Average Marks (out of 100)", f"{avg}")

    st.markdown("---")
    st.markdown("### 🏫 Engagement & Finances")
    attendance = st.slider("Attendance Percentage (%)", 0, 100, 72)
    arrears    = st.number_input("Previous Arrears (Backlogs)", 0, 10, 1, step=1)
    fees       = st.selectbox("Fees Payment Status", ["Paid", "Partially Paid", "Not Paid"])

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Risk & Generate LLM Recommendations")


# ── Helper: Encode input for ML Model ─────────────────────────────────────────
def encode_input():
    row = {
        'Age'               : age,
        'Gender'            : gender,
        'Department'        : department,
        'Residence'         : residence,
        'Guardian'          : guardian,
        'Family_Income'     : family_inc,
        'CAT1_Marks'        : cat1,
        'CAT2_Marks'        : cat2,
        'Model_Exam_Marks'  : model_marks,
        'Average_Marks'     : avg,
        'Attendance_Percent': attendance,
        'Previous_Arrears'  : int(arrears),
        'Fees_Status'       : fees,
        'Internet_Access'   : internet,
    }
    df_in = pd.DataFrame([row])
    for col, le in cat_encoders.items():
        if col in df_in.columns:
            val = df_in[col].iloc[0]
            if val in le.classes_:
                df_in[col] = le.transform([val])
            else:
                df_in[col] = 0
    return df_in[FEATURES]


# ── Helper: Fallback Rule-Based Recommendations ───────────────────────────────
def get_rule_recommendations():
    recs = []
    if cat1 < 25 or cat2 < 25:
        recs.append(("academic",
            "📖 <b>Low CAT Scores Detected:</b> Attend department peer-assisted learning and review fundamental concepts on NPTEL for core courses."))
    if model_marks < 50:
        recs.append(("academic",
            "📝 <b>Model Exam Below Passing Benchmark:</b> Solve the past 5 years' university semester papers under strict exam-timed conditions."))
    if attendance < 60:
        recs.append(("behavioral",
            "🚨 <b>Critical Attendance Shortage (&lt;60%):</b> Immediate advisor intervention required. Below 75% will block semester examination eligibility."))
    elif attendance < 75:
        recs.append(("behavioral",
            "⚠️ <b>Borderline Attendance (60–75%):</b> Target 100% attendance in the upcoming 4 weeks to regain compliance."))
    if arrears >= 3:
        recs.append(("academic",
            "📚 <b>Heavy Backlog Burden:</b> Create a dedicated 2-hour daily revision plan strictly prioritizing the oldest backlog subject."))
    elif arrears >= 1:
        recs.append(("academic",
            "📌 <b>Active Backlogs:</b> Dedicate weekend hours to backlog subjects alongside regular semester coursework."))
    if fees == 'Not Paid':
        recs.append(("financial",
            "💸 <b>Tuition Fee Pending:</b> Contact the Student Welfare Office for installment plans or emergency scholarship assistance."))
    elif fees == 'Partially Paid':
        recs.append(("financial",
            "💰 <b>Partial Fee Pending:</b> Clear the remaining balance before the exam registration deadline to avoid hall-ticket holds."))
    if internet == 'No':
        recs.append(("behavioral",
            "🌐 <b>No Home Internet:</b> Use campus computer labs to download offline lecture notes, textbooks, and NPTEL videos."))
    if not recs:
        recs.append(("behavioral",
            "🌟 <b>Strong Standing:</b> Maintain this momentum! Engage in department research projects, hackathons, and internship preparation."))
    return recs


# ── Helper: Groq LLM Recommendation Engine ───────────────────────────────────
def generate_llm_recommendations(groq_key, model_choice, profile_summary, risk_level, confidence):
    client = get_groq_client(groq_key)
    if not client:
        return None, "Groq client could not be initialized. Please check your API key."

    prompt = f"""
You are an expert Academic Dean, Student Success Director, and AI Educational Mentor.
A student has been evaluated by our Machine Learning Risk Prediction Model.

### Student Profile:
- **Department**: {profile_summary['Department']} (Age: {profile_summary['Age']}, Gender: {profile_summary['Gender']})
- **Residence**: {profile_summary['Residence']} | **Guardian**: {profile_summary['Guardian']}
- **Socio-Economic**: Family Income: {profile_summary['Family_Income']} | Internet at Home: {profile_summary['Internet_Access']} | Fees: {profile_summary['Fees_Status']}
- **Academic Performance**:
  - CAT 1: {profile_summary['CAT1_Marks']}/50
  - CAT 2: {profile_summary['CAT2_Marks']}/50
  - Model Exam: {profile_summary['Model_Exam_Marks']}/100
  - Weighted Average: {profile_summary['Average_Marks']}/100
- **Attendance**: {profile_summary['Attendance_Percent']}%
- **Arrears (Backlogs)**: {profile_summary['Previous_Arrears']}
- **Predicted Risk Level**: {risk_level} (Model Confidence: {confidence*100:.1f}%)

### Instructions:
Generate a personalized, empathetic, highly actionable, and structured Academic Recovery & Learning Plan.
Format your response in clean, professional GitHub-flavored Markdown using the following distinct sections:

1. 🎯 **Root Cause Analysis & Assessment Summary**
   - Directly analyze the primary risk factors (e.g., marks trend, attendance deficit, arrears, fees, internet barriers).
2. 🗓️ **4-Week Structured Action Plan**
   - Provide a realistic week-by-week roadmap (Week 1: Stabilization, Week 2: Core Concept Mastery, Week 3: Practice & Problem Solving, Week 4: Mock Assessments).
3. 📚 **Subject & Skill Mastery Recommendations**
   - Specific study strategies tailored to the **{profile_summary['Department']}** curriculum.
   - Exact recommendations for clearing backlogs and boosting CAT/Model scores.
4. 💡 **Resource & Welfare Navigation**
   - Specific recommendations for addressing the student's constraints (e.g. offline study tips if no internet, fee installment/scholarship navigation, peer tutoring).
   - Curated platforms (NPTEL, SWAYAM, GeeksforGeeks, MIT OCW, Department Labs).
5. 💬 **Motivational Guidance & Mentor Note**
   - A brief, encouraging message from the mentor to boost confidence.

Tone: Professional, supportive, constructive, and highly practical. Avoid generic platitudes; make the recommendations concrete.
"""

    try:
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class academic advisor and university mentor who writes comprehensive, highly structured student improvement plans in clear Markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.6,
            max_tokens=1500
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, str(e)


# ── Top Snapshot Metric Cards ─────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>CAT 1 Score</div>
        <div class='metric-value'>{cat1:.1f}<span style='font-size:16px;color:#94a3b8;font-weight:500'> / 50</span></div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>CAT 2 Score</div>
        <div class='metric-value'>{cat2:.1f}<span style='font-size:16px;color:#94a3b8;font-weight:500'> / 50</span></div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>Model Exam</div>
        <div class='metric-value'>{model_marks:.1f}<span style='font-size:16px;color:#94a3b8;font-weight:500'> / 100</span></div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""<div class='card'>
        <div class='card-title'>Average Marks</div>
        <div class='metric-value'>{avg:.2f}<span style='font-size:16px;color:#94a3b8;font-weight:500'> / 100</span></div>
    </div>""", unsafe_allow_html=True)


# ── Prediction & Recommendation Execution ─────────────────────────────────────
if predict_btn:
    with st.spinner("🤖 Running Random Forest ML Classifier..."):
        X_in       = encode_input()
        pred_enc   = model.predict(X_in)[0]
        pred_label = label_enc.inverse_transform([pred_enc])[0]
        proba      = model.predict_proba(X_in)[0]
        prob_dict  = {label_enc.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
        conf       = prob_dict[pred_label]

    profile_dict = {
        'Department': department, 'Age': age, 'Gender': gender,
        'Residence': residence, 'Guardian': guardian, 'Family_Income': family_inc,
        'Internet_Access': internet, 'Fees_Status': fees,
        'CAT1_Marks': cat1, 'CAT2_Marks': cat2, 'Model_Exam_Marks': model_marks,
        'Average_Marks': avg, 'Attendance_Percent': attendance,
        'Previous_Arrears': arrears
    }

    # Store in session state for follow-up interaction
    st.session_state['last_prediction'] = {
        'label': pred_label,
        'conf': conf,
        'prob_dict': prob_dict,
        'profile': profile_dict
    }

# ── Render Results if Available ───────────────────────────────────────────────
if 'last_prediction' in st.session_state:
    res = st.session_state['last_prediction']
    pred_label = res['label']
    conf = res['conf']
    prob_dict = res['prob_dict']
    profile_dict = res['profile']

    st.markdown("---")

    # Layout: Prediction Outcome Card & Probability Breakdown Chart
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("### 🎯 Machine Learning Prediction")
        badge_cls = {
            "High Risk": "badge-high",
            "Medium Risk": "badge-medium",
            "Low Risk": "badge-low"
        }[pred_label]
        emoji = {"High Risk": "⚠️", "Medium Risk": "🟡", "Low Risk": "✅"}[pred_label]

        st.markdown(f"""
        <div class='card' style='text-align:center; padding:30px;'>
            <div style='font-size:46px; margin-bottom:8px;'>{emoji}</div>
            <div>
                <span class='{badge_cls}'>{pred_label}</span>
            </div>
            <div style='color:#64748b; font-size:15px; margin-top:14px;'>
                Classification Confidence: <b>{conf*100:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        attend_color = "#ef4444" if profile_dict['Attendance_Percent'] < 75 else "#10b981"
        fee_color = "#ef4444" if profile_dict['Fees_Status'] == "Not Paid" else ("#f59e0b" if profile_dict['Fees_Status'] == "Partially Paid" else "#10b981")

        st.markdown(f"""
        <div class='card' style='padding:16px 20px;'>
            <table style='width:100%; border-collapse:collapse; font-size:14px;'>
                <tr style='border-bottom:1px solid #f1f5f9; height:34px;'>
                    <td style='color:#64748b; font-weight:600;'>Department</td>
                    <td style='text-align:right; font-weight:600; color:#0f172a;'>{profile_dict['Department']}</td>
                </tr>
                <tr style='border-bottom:1px solid #f1f5f9; height:34px;'>
                    <td style='color:#64748b; font-weight:600;'>Attendance</td>
                    <td style='text-align:right; font-weight:700; color:{attend_color};'>{profile_dict['Attendance_Percent']}%</td>
                </tr>
                <tr style='border-bottom:1px solid #f1f5f9; height:34px;'>
                    <td style='color:#64748b; font-weight:600;'>Fees Status</td>
                    <td style='text-align:right; font-weight:700; color:{fee_color};'>{profile_dict['Fees_Status']}</td>
                </tr>
                <tr style='border-bottom:1px solid #f1f5f9; height:34px;'>
                    <td style='color:#64748b; font-weight:600;'>Active Arrears</td>
                    <td style='text-align:right; font-weight:600; color:#0f172a;'>{profile_dict['Previous_Arrears']} subject(s)</td>
                </tr>
                <tr style='height:34px;'>
                    <td style='color:#64748b; font-weight:600;'>Internet Access</td>
                    <td style='text-align:right; font-weight:600; color:#0f172a;'>{profile_dict['Internet_Access']}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        st.markdown("### 📊 Multi-Class Risk Probabilities")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(6, 3.2))
        labels  = ['Low Risk', 'Medium Risk', 'High Risk']
        values  = [prob_dict.get('Low Risk', 0),
                   prob_dict.get('Medium Risk', 0),
                   prob_dict.get('High Risk', 0)]
        colors  = ['#22c55e', '#f59e0b', '#ef4444']

        bars = ax.barh(labels, values, color=colors, height=0.45, edgecolor='none')
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f"{val*100:.1f}%", va='center', fontsize=11, fontweight='bold', color='#1e293b')

        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Probability Score", fontsize=10, color='#64748b')
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(left=False, labelsize=11)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Recommendations Section (Tabs: Groq LLM & Quick Alerts) ───────────────
    st.markdown("---")
    st.markdown("## 🧠 Personalised AI Recommendations")

    tab1, tab2, tab3 = st.tabs([
        "✨ LLM Academic Mentor (Groq Qwen 27B)",
        "📋 Quick Intervention Checklist",
        "💬 Interactive Q&A with AI Mentor"
    ])

    with tab1:
        st.markdown("Generates a comprehensive, tailored improvement strategy using high-speed Groq LLM inference:")

        # Generate or load cached LLM plan
        cache_key = f"llm_plan_{pred_label}_{profile_dict['Average_Marks']}_{profile_dict['Attendance_Percent']}_{profile_dict['Previous_Arrears']}"

        if cache_key not in st.session_state:
            with st.spinner("🤖 Consulting Groq AI Academic Mentor..."):
                plan_text, err = generate_llm_recommendations(
                    api_key, selected_model, profile_dict, pred_label, conf
                )
                if plan_text:
                    st.session_state[cache_key] = plan_text
                else:
                    st.session_state[cache_key] = None
                    st.session_state['llm_error'] = err

        llm_output = st.session_state.get(cache_key)

        if llm_output:
            st.markdown(f"""
<div class='llm-box'>

{llm_output}

</div>
""", unsafe_allow_html=True)

            # Option to re-generate or download
            st.markdown("<br>", unsafe_allow_html=True)
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    label="📥 Download Plan (.md)",
                    data=llm_output,
                    file_name=f"student_intervention_plan_{pred_label.lower().replace(' ', '_')}.md",
                    mime="text/markdown"
                )
            with col_d2:
                if st.button("🔄 Regenerate AI Plan"):
                    st.session_state.pop(cache_key, None)
                    st.rerun()
        else:
            err_msg = st.session_state.get('llm_error', 'Unknown error occurred')
            st.warning(f"⚠️ Could not generate plan via Groq LLM: {err_msg}")
            st.info("💡 You can review the rule-based intervention checklist in the next tab.")

    with tab2:
        st.markdown("### Rapid Assessment Alerts")
        rule_recs = get_rule_recommendations()
        type_label = {
            "academic": "🔵 Academic",
            "financial": "🟡 Financial",
            "behavioral": "🟢 Behavioural & Attendance"
        }
        card_class = {
            "academic": "rec-card rec-card-academic",
            "financial": "rec-card rec-card-financial",
            "behavioral": "rec-card rec-card-behavioral"
        }

        for rtype, rtxt in rule_recs:
            st.markdown(f"""
            <div class='{card_class[rtype]}'>
                <div style='font-size:11px; font-weight:700; margin-bottom:4px;
                            text-transform:uppercase; letter-spacing:.06em; opacity:.75;'>
                    {type_label[rtype]}
                </div>
                <div>{rtxt}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("### 💬 Chat with the AI Academic Mentor")
        st.markdown("Ask specific questions about this student's recovery plan, study schedule, or subject guidance:")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("E.g., How should this student prioritize their backlogs alongside current semester exams?")

        if user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            client = get_groq_client(api_key)
            if client:
                with st.chat_message("assistant"):
                    with st.spinner("AI Mentor is thinking..."):
                        try:
                            context_prompt = (
                                f"Context: Student is in {profile_dict['Department']}, {pred_label} status. "
                                f"CAT1: {profile_dict['CAT1_Marks']}/50, CAT2: {profile_dict['CAT2_Marks']}/50, "
                                f"Model: {profile_dict['Model_Exam_Marks']}/100, Attendance: {profile_dict['Attendance_Percent']}%, "
                                f"Arrears: {profile_dict['Previous_Arrears']}, Fees: {profile_dict['Fees_Status']}.\n"
                                f"User Question: {user_query}\n"
                                f"Provide an empathetic, actionable, concise response specifically for this student."
                            )
                            c_resp = client.chat.completions.create(
                                model=selected_model,
                                messages=[
                                    {"role": "system", "content": "You are a university academic counselor answering specific questions about a student's study plan."},
                                    {"role": "user", "content": context_prompt}
                                ],
                                temperature=0.6,
                                max_tokens=600
                            )
                            reply = c_resp.choices[0].message.content
                            st.markdown(reply)
                            st.session_state.chat_history.append({"role": "assistant", "content": reply})
                        except Exception as ex:
                            err_reply = f"Error querying Groq: {ex}"
                            st.error(err_reply)
                            st.session_state.chat_history.append({"role": "assistant", "content": err_reply})
            else:
                st.error("Groq client not available. Please verify the API key.")

else:
    # Default placeholder view before clicking predict
    st.markdown("""
    <div class='card' style='text-align:center; padding:50px 20px; border:2px dashed #cbd5e1;
                              background:#f8fafc; border-radius:16px;'>
        <div style='font-size:42px; margin-bottom:12px;'>👈</div>
        <div style='font-size:20px; font-weight:700; color:#1e293b;'>
            Fill in the Student Profile on the Left Sidebar
        </div>
        <div style='color:#64748b; margin-top:8px; font-size:15px; max-width:550px; margin-left:auto; margin-right:auto;'>
            Adjust the personal indicators, CAT marks, and attendance. Then click 
            <b style='color:#4f46e5;'>'Predict Risk &amp; Generate LLM Recommendations'</b> to run the machine learning model and activate the Groq AI Mentor.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; font-size:12.5px; padding:10px 0 20px 0;'>
    EduRisk AI System &nbsp;•&nbsp;
    Classifier: Random Forest (200 Estimators) &nbsp;•&nbsp;
    Inference Engine: Groq LPU (Qwen 27B) &nbsp;•&nbsp;
    Trained on 400 Student Records
</div>
""", unsafe_allow_html=True)
