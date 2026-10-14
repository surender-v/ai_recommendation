# 🎓 EduRisk AI – Student Risk Prediction & LLM Recommendation System

An end-to-end Machine Learning and Generative AI system designed for educational institutions to proactively identify students at risk of academic failure or dropout, and deliver personalized intervention plans using **Groq-powered LLMs**.

---

## 🌟 Key Features

1. **Machine Learning Risk Classifier:**
   - Multi-class classification: **Low Risk**, **Medium Risk**, **High Risk**.
   - Model: **Random Forest Classifier** trained on realistic student indicators (CAT1, CAT2, Model Exam, Attendance, Backlogs, Fees, Internet Access).
   - High classification accuracy with class probability breakdown charts.

2. **Groq LLM Personalized Recommendation Engine:**
   - Powered by **Qwen 27B** via high-speed **Groq LPU Inference**.
   - Tailored 5-part intervention blueprint:
     - 🎯 Root Cause Analysis & Assessment Summary
     - 🗓️ 4-Week Structured Action Plan
     - 📚 Department-Specific Subject & Skill Mastery
     - 💡 Welfare, Financial, and Resource Navigation
     - 💬 Motivational Guidance from AI Academic Mentor
   - Option to **download/export the plan as Markdown (`.md`)**.

3. **Interactive AI Academic Mentor (Chatbot):**
   - Live Q&A in the dashboard to ask questions directly about backlogs, study schedules, and faculty advisor interactions.

4. **Rapid Rule-Based Intervention Checklist:**
   - Color-coded instant alerts for academic, financial, and behavioral indicators.

---

## 📁 Repository Structure

```
SURE_FINAL/
├── data/
│   └── student_dataset.csv     # 400 realistic student academic & demographic records
├── models/
│   ├── risk_model.pkl          # Trained Random Forest classifier artifact
│   ├── label_encoder.pkl       # Target encoder for risk levels
│   ├── feature_columns.pkl     # Feature column list
│   └── encoders.pkl            # Categorical label encoders
├── src/
│   ├── __init__.py             # Python package initializer
│   ├── generate_dataset.py     # Correlated student dataset generator
│   └── train_model.py          # ML training and evaluation pipeline
├── .gitignore                  # Git ignore rules for Python, ML models, & cache
├── app.py                      # Main Streamlit web application & Groq LLM integration
├── requirements.txt            # Python dependencies
└── README.md                   # Comprehensive project documentation
```

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Generate Dataset & Re-train Model
If you wish to regenerate the dataset or re-train the ML model:
```bash
# Generate synthetic dataset (saved to data/student_dataset.csv)
python src/generate_dataset.py

# Train Random Forest model (artifacts saved to models/)
python src/train_model.py
```

### 3. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## 🔑 Groq LLM Configuration
The dashboard is pre-configured with your Groq API key:
- **Default Model:** `qwen/qwen3.8-27b`
- **Fallback Models:** `openai/gpt-oss-120b`, `openai/gpt-oss-20b`
- You can adjust the API key or model anytime from the **AI Settings (Groq)** expander in the sidebar.
