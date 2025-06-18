import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("🔧 Assessment Settings")
ca_total = st.sidebar.number_input("Total CA Marks", min_value=1, value=230)
exam_total = st.sidebar.number_input("Total Exam Marks", min_value=1, value=60)
ca_weight = st.sidebar.slider("CA Weight (%)", min_value=0, max_value=100, value=60)
exam_weight = 100 - ca_weight
st.sidebar.write(f"Exam Weight: {exam_weight}%")

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📁 Upload the student CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # Detect file type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    # Cleaning and preprocessing
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Calculate CA% and Exam%
    df["CA_Percent"] = (pd.to_numeric(df["CA_Marks"], errors="coerce") / ca_total) * ca_weight
    df["Exam_Percent"] = (pd.to_numeric(df["Exam_Marks"], errors="coerce") / exam_total) * exam_weight

    # Compute overall score
    def compute_overall(ca, exam):
        if str(ca).strip().upper() == "ABS" or str(exam).strip().upper() == "ABS":
            return "ABS"
        try:
            return float(ca) + float(exam)
        except:
            return None

    df["Overall"] = df.apply(lambda row: compute_overall(row["CA_Percent"], row["Exam_Percent"]), axis=1)

    # Grade assignment
    def get_grade(score):
        if str(score).strip().upper() == "ABS":
            return "ABS"
        try:
            score = float(score)
            if score >= 90: return "A*"
            elif score >= 80: return "A"
            elif score >= 70: return "B"
            elif score >= 60: return "C"
            elif score >= 50: return "D"
            elif score >= 40: return "E"
            else: return "U"
        except:
            return "U"

    df["Grade"] = df["Overall"].apply(get_grade)
    grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]
    df["Grade"] = pd.Categorical(df["Grade"], categories=grade_order, ordered=True)

    # Benchmark
    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    benchmark = numeric_overall.mean()
    df["Above_Benchmark"] = numeric_overall > benchmark

    # -----------------------
    # Data Previews
    # -----------------------
    st.subheader("📌 Preview Data")
    st.dataframe(df.head())

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df[["CA_Percent", "Exam_Percent", "Overall"]].apply(pd.to_numeric, errors='coerce').describe())

    # -----------------------
    # Gender Distribution
    # -----------------------
    st.subheader("👫 Gender Distribution")
    gender_count = df["Gender"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(gender_count.rename("Count"))
    with col2:
        st.bar_chart(gender_count)

    # -----------------------
    # Grade Distribution
    # -----------------------
    st.subheader("🎓 Grade Distribution")
    grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(grade_dist.rename("Count"))
    with col2:
        st.bar_chart(grade_dist)

    # -----------------------
    # Performance by Gender
    # -----------------------
    st.subheader("📊 Average Scores by Gender")
    st.dataframe(df.groupby("Gender")[["CA_Percent", "Exam_Percent", "Overall"]].mean())

    # -----------------------
    # Class Performance
    # -----------------------
    st.subheader("🏫 Class Performance Summary")
    st.dataframe(df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].describe().round(2))
    st.subheader("📌 Overall Mean Score by Class")
    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    df["Numeric_Overall"] = numeric_overall
    st.bar_chart(df.groupby("Class")["Numeric_Overall"].mean()) 

    # -----------------------
    # Gender-Grade Crosstab
    # -----------------------
    st.subheader("🧮 Gender-wise Grade Distribution")
    st.dataframe(df.groupby(["Gender", "Grade"]).size().unstack(fill_value=0).reindex(columns=grade_order, fill_value=0))

    # -----------------------
    # Benchmark Comparison
    # -----------------------
    st.subheader("📉 Above/Below Average Performance")
    above = df["Above_Benchmark"].sum()
    below = len(df) - above
    col1, col2 = st.columns(2)
    col1.metric("Above Benchmark", above)
    col2.metric("Below Benchmark", below)

    # -----------------------
    # Cluster Analysis
    # -----------------------
    st.subheader("🔍 Cluster Analysis (CA% vs Exam%)")
    valid_scores = df[pd.to_numeric(df["Overall"], errors="coerce").notna()]
    if not valid_scores.empty:
        kmeans = KMeans(n_clusters=3, n_init=10)
        valid_scores = valid_scores.copy()
        valid_scores["Cluster"] = kmeans.fit_predict(valid_scores[["CA_Percent", "Exam_Percent"]])
        df = df.merge(valid_scores[["CA_Percent", "Exam_Percent", "Cluster"]], on=["CA_Percent", "Exam_Percent"], how="left")

        fig, ax = plt.subplots()
        for cluster in sorted(valid_scores["Cluster"].unique()):
            cluster_df = valid_scores[valid_scores["Cluster"] == cluster]
            ax.scatter(cluster_df["CA_Percent"], cluster_df["Exam_Percent"], label=f"Cluster {int(cluster)}")
        ax.set_xlabel("CA_Percent")
        ax.set_ylabel("Exam_Percent")
        ax.set_title("Student Performance Clusters")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("❗ Not enough valid numeric data to perform clustering.")

    # -----------------------
    # Download Option
    # -----------------------
    st.download_button("📥 Download Analyzed CSV", data=df.to_csv(index=False), file_name="student_analysis_results.csv", mime="text/csv")
