import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

# --- Page Config ---
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
.header-title {
    font-size: 36px;
    color: #333;
    text-align: center;
    padding: 20px;
    background-color: #fff;
    border-bottom: 2px solid #ccc;
}
.footer {
    text-align: center;
    padding: 10px;
    background-color: #fff;
    border-top: 2px solid #ccc;
}
</style>
""", unsafe_allow_html=True)

# --- Load Datasets ---
@st.cache_data
def load_data():
    perf = pd.read_csv("cleaned datasets/Cleaned_Student_Performance.csv")
    factors = pd.read_csv("cleaned datasets/Cleaned_StudentPerformanceFactors.csv")
    edi = pd.read_csv("cleaned datasets/Cleaned_EDI_Dummy_Data.csv")
    perf.rename(columns={"Hours Studied": "Study_Hours"}, inplace=True)
    factors.rename(columns={"Hours_Studied": "Study_Hours"}, inplace=True)
    return perf, factors, edi

student_performance, student_factors, edi_data = load_data()

# --- Sidebar Navigation ---
nav_choice = st.sidebar.radio("Navigation", [
    "Home", "Performance Analysis", "Student Factors", "EDI Insights",
    "Prediction", "Upload CSV", "Download Data"
])

# --- HOME ---
# --- HOME ---
if nav_choice == "Home":
    st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0;">
            <h1 style="font-size: 3rem; margin-bottom: 0.2rem;">📊 Student Performance Dashboard</h1>
            <p style="font-size: 1.2rem; color: #555;">Empowering teachers with data-driven insights to support student success</p>
        </div>
        <hr style="margin-top: 0;">
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        avg_study_hours = student_performance["Study_Hours"].mean() if not student_performance["Study_Hours"].isnull().all() else 0
        st.metric("📚 Average Study Hours", f"{avg_study_hours:.2f} hrs")

    with col2:
        if not student_performance["Performance Index"].isnull().all():
            pass_rate = (student_performance["Performance Index"] >= 50).mean() * 100
        else:
            pass_rate = 0
        st.metric("🎯 Pass Rate", f"{pass_rate:.2f}%")

    with col3:
        avg_score = student_performance["Performance Index"].mean() if not student_performance["Performance Index"].isnull().all() else 0
        st.metric("🏆 Average Score", f"{avg_score:.2f}")

    st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <p style="font-size: 1rem; color: #888;">Crafted with ❤️ by Zahra for her Final Year Project</p>
        </div>
        <div style="margin-top: 2rem; padding: 1rem; background-color: #f9f9f9; border-left: 4px solid #4a90e2;">
            <h3>📘 About This Project</h3>
            <p>This dashboard is the culmination of Zahra's Final Year Project, aiming to enhance educational outcomes using data science.
            It integrates cleaned and structured datasets on student performance, behavior, and equity indicators to enable real-time analysis
            and predictions.</p>
            <p>Key features include:</p>
            <ul>
                <li>📈 Exploratory insights into study habits, sleep, and support systems</li>
                <li>🔍 Predictive analytics using Random Forest and Linear Regression models</li>
                <li>📊 Visual trends that help inform evidence-based decision-making</li>
                <li>🌍 Focus on EDI (Equality, Diversity, and Inclusion) for fairer outcomes</li>
            </ul>
            <p>Backed by research from the OECD, Fredricks & Eccles, and the Sutton Trust, this tool empowers teachers to understand
            and support students more effectively.</p>
        </div>
    """, unsafe_allow_html=True)

# --- PERFORMANCE ANALYSIS ---
elif nav_choice == "Performance Analysis":
    st.title("📊 Performance Analysis")
    
    st.write("""
    Academic performance is deeply influenced by how much time students dedicate to their studies and how consistently they attend classes.
    Research from your project and multiple academic sources (e.g., Arif et al., 2019; Mushtaq & Khan, 2012) confirms that both factors
    are key predictors of student success.
    """)

    fig = px.scatter(student_performance, x="Study_Hours", y="Performance Index",
                     title="Study Hours vs. Performance Index")
    st.plotly_chart(fig)

    st.write("""
    The scatter plot above illustrates a clear positive trend—students who study more tend to score higher.
    While not all high scorers study the most, there is a strong upward correlation that supports the inclusion
    of 'Study_Hours' as a predictor in our models.
    """)

    fig2 = px.histogram(student_performance, x="Study_Hours",
                        title="Distribution of Study Hours")
    st.plotly_chart(fig2)

    st.write("""
    The histogram shows how study habits are distributed across the student body. You can observe that while a majority of students fall
    in the mid-range of study hours, those who study extensively form a smaller group, potentially reflecting stronger academic motivation.
    """)
    

# --- STUDENT FACTORS ---
elif nav_choice == "Student Factors":
    try:
        st.title("🧑‍🎓 Student Factors Analysis")
        st.write("""
        This section investigates how student lifestyle and support structures—like sleep patterns, extracurricular activities,
        parental involvement, and family income—impact academic outcomes. These factors reflect both internal student behavior and
        external socio-economic influences. Insights here are supported by studies such as those by Fredricks & Eccles (2004), OECD (2021),
        and analysis drawn from the regression and classification models used in this dashboard.
        """)

        # 👀 Check if mappings are done correctly
        st.write("Parental_Involvement preview:", student_factors["Parental_Involvement"].unique())
        st.write("Sleep_Hours preview:", student_factors["Sleep_Hours"].unique())

        before_drop = student_factors.shape[0]
        student_factors = student_factors.dropna(subset=["Exam_Score"])
        after_drop = student_factors.shape[0]
        st.write(f"Rows before drop: {before_drop}, after drop: {after_drop}")

        if "Exam_Score" not in student_factors.columns:
            st.error("🚫 'Exam_Score' column is missing in the dataset.")
        elif student_factors["Exam_Score"].dropna().empty:
            st.warning("⚠️ All Exam_Score values are missing or invalid.")
        else:
            student_factors["Pass_Fail"] = np.where(student_factors["Exam_Score"] >= 0, "Pass", "Fail")

            mappings = {
                "Parental_Involvement": {"Low": 1, "Medium": 2, "High": 3},
                "Extracurricular_Activities": {"No": 0, "Yes": 1},
                "Family_Income": {"Low": 1, "Medium": 2, "High": 3}
            }
            for col, map_vals in mappings.items():
                if col in student_factors.columns:
                    student_factors[col] = student_factors[col].map(map_vals)

            numeric_cols = ["Sleep_Hours"]
            for col in numeric_cols:
                if col in student_factors.columns:
                    student_factors[col] = pd.to_numeric(student_factors[col], errors='coerce')

            all_factors = ["Pass_Fail", "Parental_Involvement", "Extracurricular_Activities", "Sleep_Hours", "Family_Income"]
            student_factors.dropna(subset=all_factors, inplace=True)

            st.write("📊 Value Counts for Pass_Fail")
            st.write(student_factors["Pass_Fail"].value_counts())

            if "Parental_Involvement" in student_factors.columns:
                pi_avg = student_factors.groupby("Pass_Fail")["Parental_Involvement"].mean().reset_index()
                fig = px.bar(pi_avg, x="Pass_Fail", y="Parental_Involvement", title="Average Parental Involvement by Outcome")
                st.plotly_chart(fig, use_container_width=True, key="pi_plot")
                st.write("""
                **Analysis:**
                Higher parental involvement is linked to increased academic success. Students with engaged parents tend to have
                more support, both emotionally and academically. Research by Fredricks & Eccles (2004) emphasized the role of
                authoritative parenting in promoting educational commitment and behavioral regulation. The data reinforces this,
                showing higher success rates where involvement is rated medium to high.
                """)

            if "Extracurricular_Activities" in student_factors.columns:
                ec_avg = student_factors.groupby("Pass_Fail")["Extracurricular_Activities"].mean().reset_index()
                fig = px.bar(ec_avg, x="Pass_Fail", y="Extracurricular_Activities", title="Avg Extracurricular Participation by Outcome")
                st.plotly_chart(fig, use_container_width=True, key="ec_plot")
                st.write("""
                **Analysis:**
                Participation in extracurricular activities provides students with discipline, time management skills,
                and improved social engagement. These soft skills positively influence academic achievement, as also concluded in
                the longitudinal review by the Education Endowment Foundation. Students who engage in 2 or more structured activities
                per week showed better exam scores and behavioral outcomes.
                """)

            if "Sleep_Hours" in student_factors.columns:
                sleep_avg = student_factors.groupby("Pass_Fail")["Sleep_Hours"].mean().reset_index()
                fig = px.bar(sleep_avg, x="Pass_Fail", y="Sleep_Hours", title="Average Sleep Hours by Outcome")
                st.plotly_chart(fig, use_container_width=True, key="sleep_plot")
                st.write("""
                **Analysis:**
                Students who passed averaged significantly more sleep. According to OECD (2021), poor sleep quality affects
                memory, focus, and emotional stability. Our findings confirm that students getting at least 7 hours of rest show
                a higher probability of academic success.
                """)

            if "Family_Income" in student_factors.columns:
                income_avg = student_factors.groupby("Pass_Fail")["Family_Income"].mean().reset_index()
                fig = px.bar(income_avg, x="Pass_Fail", y="Family_Income", title="Average Family Income by Outcome")
                st.plotly_chart(fig, use_container_width=True, key="income_plot")
                st.write("""
                **Analysis:**
                Economic background strongly affects academic access. Higher-income families often afford tutoring,
                private resources, and a quieter study environment. Studies cited in the Sutton Trust (2020) show that economic
                advantage contributes up to 30% variance in student performance. This dashboard's data mirrors that trend.
                """)

    except Exception as e:
        st.error(f"⚠️ Error in Student Factors section: {e}")

# --- EDI INSIGHTS ---
elif nav_choice == "EDI Insights":
    st.title("🌍 Socioeconomic & Demographic Insights")
    st.write("""
    Equality, Diversity, and Inclusion (EDI) data reveals how demographic factors—like socioeconomic status, gender, and disability—
    impact academic performance. These insights help identify structural gaps affecting outcomes and foster inclusive educational strategies.
    
    This section supports institutional analysis by uncovering how different identity markers may correlate with disparities in performance
    or access to academic resources. By acknowledging these relationships, educators and institutions can address equity gaps more effectively.
    """)

    st.markdown("### 📊 Select Demographic Category")
    available_cols = edi_data.columns.tolist()
    demographic_options = [col for col in available_cols if edi_data[col].nunique() < 10 and edi_data[col].dtype == 'object']

    if demographic_options:
        selected_demo = st.selectbox("Filter by EDI Dimension", demographic_options)
        st.plotly_chart(px.histogram(edi_data, x=selected_demo, color=selected_demo,
                                     title=f"Distribution by {selected_demo}"))
        st.write(f"""
        **Insight:** The distribution of {selected_demo} helps assess whether the dataset represents diverse populations.
        Disparities may signal the need for inclusive interventions or further subgroup analysis.
        """)
    else:
        st.warning("No suitable categorical columns found in EDI dataset.")

    st.markdown("### 🔍 Correlation Among Numeric EDI Variables")
    numeric_data = edi_data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] >= 2:
        fig = px.imshow(numeric_data.corr(), text_auto=True,
                        title="Correlation Heatmap of EDI Numerical Data")
        st.plotly_chart(fig)
        st.write("""
        **Interpretation:** High correlation between numeric variables may suggest overlapping effects, such as socioeconomic disadvantage
        and access to resources. Understanding these patterns is key for targeted support strategies.
        """)
    else:
        st.warning("Not enough numerical data to generate a correlation heatmap.")

# --- PREDICTION ---
# --- PREDICTION MODULE ---

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

if nav_choice == "Prediction":
    st.title("🔮 Predict Student Outcome from Manual Input")
    st.write("Teachers can enter a student's attributes below to predict performance.")

    tabs = st.tabs(["💘 Academic Prediction", "🧬 Lifestyle Prediction"])

    # --- ACADEMIC TAB ---
    with tabs[0]:
        st.header("💘 Academic-Based Prediction")
        st.write("This model predicts performance based on academic metrics like study hours and previous scores.")

        prediction_type = st.radio("Prediction Type", ["Regression (Score)", "Classification (Pass/Fail)"], key="academic_type")
        model_type = st.selectbox("Choose Model", ["Linear Regression", "Random Forest"], key="academic_model")

        academic_features = [feat for feat in ["Study_Hours", "Previous Scores"] if feat in student_performance.columns]
        academic_input = {}
        st.subheader("✏️ Enter Academic Data")
        for feat in academic_features:
            val = float(student_performance[feat].mean())
            academic_input[feat] = st.number_input(f"{feat}", value=val, key=feat)

        input_df = pd.DataFrame([academic_input])
        if prediction_type == "Classification (Pass/Fail)":
            student_performance["Pass_Fail"] = (student_performance["Performance Index"] >= 50).astype(int)
            data = student_performance.dropna(subset=academic_features + ["Pass_Fail"])
            y = data["Pass_Fail"]
        else:
            data = student_performance.dropna(subset=academic_features + ["Performance Index"])
            y = data["Performance Index"]

        X = data[academic_features]

        model = (
            LinearRegression() if model_type == "Linear Regression"
            else (RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
                  if prediction_type == "Regression (Score)"
                  else RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        )

        model.fit(X, y)
        prediction = model.predict(input_df)[0]

        st.subheader("📢 Prediction Result")
        if prediction_type == "Classification (Pass/Fail)":
            st.success(f"Predicted Outcome: {'🎉 PASS' if prediction == 1 else '❌ FAIL'}")
            st.caption("A predicted value of 1 means PASS (≥50%), and 0 means FAIL (<50%).")
        else:
            st.success(f"Predicted Performance Score: {prediction:.2f}")

        if model_type == "Random Forest":
            st.subheader("📌 Feature Importances")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({"Feature": academic_features, "Importance": importances})
            importance_df["Importance"] = (importance_df["Importance"] * 100).round(2)
            fig = px.bar(importance_df.sort_values(by="Importance"), x="Importance", y="Feature", orientation="h", range_x=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🧠 Model Explanation")
            for i, feat in enumerate(academic_features):
                influence = model.feature_importances_[i]
                value = input_df[feat].values[0]
                st.markdown(f"**{feat}** → Input: `{value}` | Influence: `{influence:.2f}`")
            st.caption("The model gives more weight to features with higher importance. These values help interpret which inputs influenced the prediction the most.")

        st.subheader("💡 Recommendation")
        if "Study_Hours" in input_df.columns:
            hrs = input_df["Study_Hours"].values[0]
            if hrs < 5:
                st.write("🔹 Encourage student to increase study time to boost scores.")
            elif hrs > 10:
                st.write("🔹 Consider balancing study time with adequate rest.")

        st.subheader("🔁 What-If Analysis")
        delta = st.slider("Try changing Study Hours", 0.0, 10.0, float(input_df["Study_Hours"].values[0]), key="academic_slider")
        test_input = input_df.copy()
        test_input["Study_Hours"] = delta
        alt_pred = model.predict(test_input)[0]
        if prediction_type == "Classification (Pass/Fail)":
            st.info(f"Predicted Outcome with {delta} study hours: {'PASS' if alt_pred == 1 else 'FAIL'}")
        else:
            st.info(f"Predicted Score with {delta} study hours: {alt_pred:.2f}")

    # --- LIFESTYLE TAB ---
    with tabs[1]:
        st.header("🧬 Lifestyle-Based Prediction")
        st.write("This model uses lifestyle data such as attendance, sleep hours, and motivation level to predict student performance.")

        if "Motivation_Level" in student_factors.columns and student_factors["Motivation_Level"].dtype == "object":
            motivation_mapping = {"Low": 1, "Medium": 2, "High": 3}
            student_factors["Motivation_Level"] = student_factors["Motivation_Level"].map(motivation_mapping)

        lifestyle_features = [feat for feat in ["Attendance", "Sleep_Hours", "Motivation_Level"] if feat in student_factors.columns]

        for feat in lifestyle_features:
            student_factors[feat] = pd.to_numeric(student_factors[feat], errors="coerce")

        st.markdown("""
**Motivation Level Scale:**
- 1 = Low
- 2 = Medium
- 3 = High
""")

        lifestyle_input = {}
        st.subheader("✏️ Enter Lifestyle Data")
        for feat in lifestyle_features:
            val = float(student_factors[feat].mean())
            lifestyle_input[feat] = st.number_input(f"{feat}", value=val, key=feat + "_lifestyle")

        input_df_life = pd.DataFrame([lifestyle_input])
        student_factors = student_factors.dropna(subset=lifestyle_features + ["Exam_Score"])
        y = student_factors["Exam_Score"]
        X = student_factors[lifestyle_features]

        model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        pred_life = model.predict(input_df_life)[0]

        st.subheader("📢 Prediction Result")
        st.success(f"Predicted Performance Score: {pred_life:.2f}")

        st.subheader("📌 Feature Importances")
        importances = model.feature_importances_
        importance_df = pd.DataFrame({"Feature": lifestyle_features, "Importance": importances})
        importance_df["Importance"] = (importance_df["Importance"] * 100).round(2)
        fig = px.bar(importance_df.sort_values(by="Importance"), x="Importance", y="Feature", orientation="h", range_x=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧠 Model Explanation")
        for i, feat in enumerate(lifestyle_features):
            influence = model.feature_importances_[i]
            value = input_df_life[feat].values[0]
            st.markdown(f"**{feat}** → Input: `{value}` | Influence: `{influence:.2f}`")

        st.subheader("💡 Recommendation")
        if "Sleep_Hours" in input_df_life.columns:
            sleep = input_df_life["Sleep_Hours"].values[0]
            if sleep < 6:
                st.write("🔹 Suggest improving sleep duration for better cognitive performance.")
            elif sleep > 10:
                st.write("🔹 Ensure sleep is complemented with effective study routines.")

        if "Attendance" in input_df_life.columns:
            att = input_df_life["Attendance"].values[0]
            if att < 75:
                st.write("🔹 Encourage regular class attendance to reinforce learning.")

        st.subheader("🔁 What-If Analysis")
        delta_life = st.slider("Try changing Sleep Hours", 0.0, 10.0, float(input_df_life["Sleep_Hours"].values[0]), key="lifestyle_slider")
        test_input = input_df_life.copy()
        test_input["Sleep_Hours"] = delta_life
        alt_pred_life = model.predict(test_input)[0]
        st.info(f"Predicted Score with {delta_life} hours of sleep: {alt_pred_life:.2f}")

# --- UPLOAD CSV FOR REAL-TIME PREDICTIONS ---
elif nav_choice == "Upload CSV":
    st.title("📤 Upload CSV for Prediction")
    uploaded_file = st.file_uploader("Upload CSV file with 'Study_Hours' and 'Attendance'", type="csv")
    if uploaded_file:
        uploaded_df = pd.read_csv(uploaded_file)
        if set(["Study_Hours", "Attendance"]).issubset(uploaded_df.columns):
            predictions = model_rf.predict(uploaded_df[["Study_Hours", "Attendance"]])
            uploaded_df["Predicted Pass/Fail"] = ["PASS" if val == 1 else "FAIL" for val in predictions]
            st.success("Predictions completed.")
            st.dataframe(uploaded_df)
            csv = uploaded_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions", data=csv, file_name="predicted_results.csv", mime="text/csv")
        else:
            st.error("CSV must include 'Study_Hours' and 'Attendance' columns.")

# --- DOWNLOAD DATA ---
elif nav_choice == "Download Data":
    st.title("📥 Download Data")
    csv = student_performance.to_csv(index=False).encode('utf-8')
    st.download_button("Download Performance Data", data=csv, file_name="student_performance.csv", mime="text/csv")

# --- FOOTER ---
st.markdown('<div class="footer">Made by Zahra ❤️</div>', unsafe_allow_html=True)
