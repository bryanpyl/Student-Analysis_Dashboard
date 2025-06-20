import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fpdf import FPDF
import tempfile
import os

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
    - **Grade**: Student's grade (A*, A, B, C, D, E.......)  

    **Example Column Headers**:
    ```
    Name, Gender, Class, CA_Marks, CA_Percent, Exam_Marks, Exam_Percent, Total, Grade
    ```

    **Note**:
    - The column names are **case-insensitive**, but ensure they follow the exact format.
    - Avoid extra spaces in headers or leave cells empty.
    - 'ABS' will be treated as **absent** and excluded from numerical computations.
    """)

# -----------------------
# File Upload
# -----------------------
uploaded_file = st.file_uploader("📁 Upload the student CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        # Read all sheets into a dictionary
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)

        # Convert each sheet into a DataFrame and tag with the sheet name as class if 'Class' is missing
        dataframes = []
        for sheet_name, sheet_df in all_sheets.items():
            sheet_df.columns = sheet_df.columns.str.strip().str.replace(" ", "_")
            if "Class" not in sheet_df.columns:
                sheet_df["Class"] = sheet_name
            dataframes.append(sheet_df)

        # Combine all sheets into a single DataFrame
        df = pd.concat(dataframes, ignore_index=True)
    
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df["CA_Percent"] = pd.to_numeric(df["CA_Percent"], errors="coerce")
    df["Exam_Percent"] = pd.to_numeric(df["Exam_Percent"], errors="coerce")  

   # Use 'Total' column directly as Overall (respect ABS if present)
    df["Overall"] = df["Total"].apply(lambda x: "ABS" if str(x).strip().upper() == "ABS" else pd.to_numeric(x, errors="coerce"))

    df["Grade"] = df["Grade"].str.strip().str.upper()
    # Define the valid grade categories
    grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]
    # Convert to categorical with proper ordering
    df["Grade"] = pd.Categorical(df["Grade"], categories=grade_order, ordered=True)

    # Calculate numeric overall for benchmark (if needed)
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
    # Mean Scores Class
    # -----------------------
    st.subheader("🏫 Class Performance Summary")
    st.dataframe(df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].describe().round(2))

    st.subheader("📌 Overall Mean Score by Class")
    df["Numeric_Overall"] = numeric_overall
    st.bar_chart(df.groupby("Class")["Numeric_Overall"].mean())
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
    # -----------------------
    # PDF Report Generation
    # -----------------------

    if st.button("🖨️ Generate PDF Report"):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save charts
            grade_by_class_path = os.path.join(tmpdirname, "grade_by_class.png")
            fig1.savefig(grade_by_class_path, bbox_inches='tight')

            overall_grade_pie_path = os.path.join(tmpdirname, "overall_grade_pie.png")
            fig2.savefig(overall_grade_pie_path, bbox_inches='tight')

            cluster_path = os.path.join(tmpdirname, "cluster_plot.png")
            if 'fig' in locals():
                fig.savefig(cluster_path, bbox_inches='tight')

            # Generate Summary Tables
            grade_counts = df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)
            grade_counts["Total"] = grade_counts.sum()
            grade_count_by_class = df.pivot_table(index="Class", columns="Grade", aggfunc='size', fill_value=0).reindex(columns=grade_order, fill_value=0)
            grade_count_by_class["Total"] = grade_count_by_class.sum(axis=1)
            class_summary = df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].mean().round(2)
            descriptive_stats = df[["CA_Percent", "Exam_Percent", "Overall"]].apply(pd.to_numeric, errors='coerce').describe().round(2)

            # Create PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, "Student Performance Report", ln=True, align="C")

            pdf.set_font("Arial", '', 12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Total Students: {len(df)}", ln=True)
            pdf.cell(0, 10, f"Mean Benchmark Score: {benchmark:.2f}", ln=True)
            pdf.cell(0, 10, f"Above Benchmark: {above}", ln=True)
            pdf.cell(0, 10, f"Below Benchmark: {below}", ln=True)

            # Grade Distribution Table
            pdf.ln(8)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Grade Count Summary by Class", ln=True)

            # Set column width based on page size
            usable_width = 190  # A4 width minus default 10mm margins on both sides
            num_columns = len(grade_order) + 2  # Grades + Class + Total
            col_width = usable_width / num_columns

            # Table header
            pdf.set_font("Arial", 'B', 9)
            pdf.cell(col_width, 8, "Class", 1)
            for grade in grade_order:
                pdf.cell(col_width, 8, grade, 1)
            pdf.cell(col_width, 8, "Total", 1)
            pdf.ln()

            # Table rows
            pdf.set_font("Arial", '', 8)
            for class_name, row in grade_count_by_class.iterrows():
                pdf.cell(col_width, 8, str(class_name), 1)
                for grade in grade_order:
                    pdf.cell(col_width, 8, str(row[grade]), 1)
                pdf.cell(col_width, 8, str(row["Total"]), 1)
                pdf.ln()

            # Class Performance Summary Table
            pdf.ln(8)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Class Summary (Average CA%, Exam%, Overall)", ln=True)
            pdf.set_font("Arial", '', 9)
            col_width = 40
            pdf.cell(col_width, 8, "Class", 1)
            for col in class_summary.columns:
                pdf.cell(col_width, 8, col, 1)
            pdf.ln()
            for idx, row in class_summary.iterrows():
                pdf.cell(col_width, 8, str(idx), 1)
                for val in row:
                    pdf.cell(col_width, 8, str(val), 1)
                pdf.ln()

            # Descriptive Statistics
            pdf.ln(8)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Descriptive Statistics", ln=True)
            pdf.set_font("Arial", '', 9)
            pdf.cell(col_width, 8, "Metric", 1)
            for col in descriptive_stats.columns:
                pdf.cell(col_width, 8, col, 1)
            pdf.ln()
            for idx, row in descriptive_stats.iterrows():
                pdf.cell(col_width, 8, str(idx), 1)
                for val in row:
                    pdf.cell(col_width, 8, str(val), 1)
                pdf.ln()

            # Charts
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Grade Distribution by Class", ln=True)
            pdf.image(grade_by_class_path, w=180)

            pdf.ln(10)
            pdf.image(overall_grade_pie_path, w=150)

            if os.path.exists(cluster_path):
                pdf.add_page()
                pdf.cell(0, 10, "Performance Cluster Chart", ln=True)
                pdf.image(cluster_path, w=180)

            # Save and Download
            pdf_output_path = os.path.join(tmpdirname, "Student_Performance_Report.pdf")
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as f:
                st.download_button("📄 Download PDF Report", data=f, file_name="Student_Performance_Report.pdf", mime="application/pdf")

        