import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# -----------------------
# Instruction Section
# -----------------------
with st.expander("📘 Instructions: Preparing Your Excel File", expanded=True):
    st.markdown("""
    Please ensure your Excel or CSV file includes the following columns:

    - **Name**: Student's full name  
    - **Gender**: e.g., Male / Female  
    - **Class**: Class name or code  
    - **CA_Marks**: Continuous Assessment marks (numeric or 'ABS' if absent) 
    - **CA_Percent**: CA percentage (numeric, will be calculated if not provided) 
    - **Exam_Marks**: Final exam marks (numeric or 'ABS' if absent)
    - **Exam_Percent**: Exam percentage (numeric, will be calculated if not provided)
    - **Total**: Total marks (numeric, will be calculated as CA_Percent + Exam_Percent)

    **Example Column Headers**:
    ```
    Name, Gender, Class, CA_Marks, CA_Percent, Exam_Marks, Exam_Percent, Total
    ```

    **Note**:
    - The column names are **case-insensitive**, but ensure they follow the exact format.
    - Avoid extra spaces in headers or leave cells empty.
    - 'ABS' will be treated as absent and excluded from numerical computations.
    """)

# -----------------------
# Sidebar Inputs
# -----------------------
st.sidebar.header("🔧 Assessment Settings")
ca_total = st.sidebar.number_input("Total CA Marks", min_value=1, value=100)
exam_total = st.sidebar.number_input("Total Exam Marks", min_value=1, value=50)
ca_weight = st.sidebar.slider("CA Weight (%)", min_value=0, max_value=100, value=50)
exam_weight = 100 - ca_weight
st.sidebar.write(f"Exam Weight: {exam_weight}%")

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📁 Upload the student CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df["CA_Percent"] = (pd.to_numeric(df["CA_Marks"], errors="coerce") / ca_total) * ca_weight
    df["Exam_Percent"] = (pd.to_numeric(df["Exam_Marks"], errors="coerce") / exam_total) * exam_weight

    def compute_overall(ca, exam):
        if str(ca).strip().upper() == "ABS" or str(exam).strip().upper() == "ABS":
            return "ABS"
        try:
            return float(ca) + float(exam)
        except:
            return None

    df["Overall"] = df.apply(lambda row: compute_overall(row["CA_Percent"], row["Exam_Percent"]), axis=1)

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

    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    benchmark = numeric_overall.mean()
    df["Above_Benchmark"] = numeric_overall > benchmark
    
    # -----------------------
    # Preview Dataset
    # -----------------------
    st.subheader("📌 Preview Data")
    st.dataframe(df.head())

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df[["CA_Percent", "Exam_Percent", "Overall"]].apply(pd.to_numeric, errors='coerce').describe())
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    # -----------------------
    # Gender-Grade Analysis
    # -----------------------
    st.subheader("👫 Gender Distribution")
    gender_count = df["Gender"].value_counts()
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(gender_count.rename("Count"))
    with col2:
        st.bar_chart(gender_count)
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    # -----------------------
    # Grade distribution (Overall, barchart and table)
    # -----------------------
    st.subheader("🎓 Grade Distribution")

    # --- Overall Grade Distribution ---
    st.markdown("#### 🧮 Overall")
    grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(grade_dist.rename("Count"))
    with col2:
        st.bar_chart(grade_dist)

    # --- Class-wise Grade Distribution ---
    st.markdown("#### 🏷️ By Class")
    classes = sorted(df["Class"].dropna().unique())

    for class_name in classes:
        with st.expander(f"📘 Class: {class_name}", expanded=False):
            class_df = df[df["Class"] == class_name]
            class_grade_dist = class_df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(class_grade_dist.rename("Count"))
            with col2:
                st.bar_chart(class_grade_dist)

    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # -----------------------
    # Mean Scores by Gender and Class
    # -----------------------
    st.subheader("📊 Average Scores by Gender")
    st.dataframe(df.groupby("Gender")[["CA_Percent", "Exam_Percent", "Overall"]].mean())

    st.subheader("🏫 Class Performance Summary")
    st.dataframe(df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].describe().round(2))

    st.subheader("📌 Overall Mean Score by Class")
    df["Numeric_Overall"] = numeric_overall
    st.bar_chart(df.groupby("Class")["Numeric_Overall"].mean())
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    # -----------------------
    # Gender-wise Grade Distribution
    # -----------------------
    st.subheader("🧮 Gender-wise Grade Distribution")
    st.dataframe(df.groupby(["Gender", "Grade"]).size().unstack(fill_value=0).reindex(columns=grade_order, fill_value=0))
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    # -----------------------
    # Benchmark Analysis (Mean)
    # -----------------------
    st.subheader("📉 Above/Below Average Performance")
    above = df["Above_Benchmark"].sum()
    below = len(df) - above
    col1, col2 = st.columns(2)
    col1.metric("Above Benchmark", above)
    col2.metric("Below Benchmark", below)
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    # -----------------------
    # Graphical Analysis
    # -----------------------
    st.subheader("📊 Graphical Analysis")

    # 1. Grade distribution by class (bar chart)
    st.markdown("### 🎯 Grade Distribution by Class")

    grade_by_class = df.groupby(['Class', 'Grade']).size().unstack(fill_value=0).reindex(columns=grade_order, fill_value=0)

    st.dataframe(grade_by_class)

    st.markdown("Bar chart showing number of students in each grade category for every class.")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    grade_by_class.plot(kind="bar", ax=ax1)
    ax1.set_ylabel("Number of Students")
    ax1.set_xlabel("Class")
    ax1.set_title("Grade Distribution by Class")
    ax1.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig1)
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # 2. Overall grade distribution (pie chart)
    st.markdown("### 📌 Overall Grade Distribution (Percentage)")

    overall_grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0)
    total_students = overall_grade_dist.sum()
    percentage_labels = [f"{g}: {int(c)} students ({c/total_students:.1%})" for g, c in overall_grade_dist.items() if c > 0]

    fig2, ax2 = plt.subplots()
    ax2.pie(overall_grade_dist[overall_grade_dist > 0], labels=percentage_labels, labeldistance=1.1, textprops={'fontsize': 12, 'fontweight': 'normal'}, autopct='%1.1f%%', startangle=140, counterclock=False)
    ax2.set_title("Overall Grade Distribution (All Classes)", fontsize=14, fontweight='bold')
    st.pyplot(fig2)

    st.markdown("Each segment of the pie chart shows the percentage and count of students falling under each grade across all classes.")
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
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

    st.download_button("📥 Download Analyzed CSV", data=df.to_csv(index=False), file_name="student_analysis_results.csv", mime="text/csv")
