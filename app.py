import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fpdf import FPDF
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# -----------------------
# PDF Generation Function
# -----------------------
def generate_pdf_report(df, benchmark, grade_dist, grade_by_class, overall_pie_path, cluster_chart_path, overall_bar_chart_path, class_bar_chart_paths, report_title="Student Performance Report"):
    from fpdf import FPDF
    import matplotlib.pyplot as plt
    import os
    import tempfile

    # Grade order consistency
    grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, report_title, ln=True, align='C')

    # ----------------- Overall Grade Distribution Charts -----------------
    pdf.set_font("Arial", 'B', 12)
    pdf.ln(3)
    pdf.cell(0, 10, "Overall Grade Distribution (Bar Chart)", ln=True)
    if os.path.exists(overall_bar_chart_path):
        pdf.image(overall_bar_chart_path, w=170)

    pdf.ln(4)
    pdf.cell(0, 10, "Overall Grade Distribution (Pie Chart)", ln=True)
    if os.path.exists(overall_pie_path):
        pdf.image(overall_pie_path, w=170)

    # ----------------- Cluster Analysis -----------------
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Cluster Analysis (CA% vs Exam%)", ln=True)
    if cluster_chart_path and os.path.exists(cluster_chart_path):
        pdf.image(cluster_chart_path, w=180)

    # ----------------- Benchmark Summary -----------------
    above = df["Above_Benchmark"].sum()
    below = len(df) - above
    pdf.ln(5)
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 10, f"Benchmark (Mean Overall Score): {benchmark:.2f}\n"
                          f"Above Benchmark: {above} students\n"
                          f"Below Benchmark: {below} students")

    # ----------------- Class-wise Grade Distributions -----------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Grade Distribution by Class", ln=True)

    for class_name, path in class_bar_chart_paths.items():
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Class: {class_name}", ln=True)
        pdf.image(path, w=140)
        pdf.ln(4)

    # ----------------- Save PDF -----------------
    pdf_path = os.path.join(tempfile.gettempdir(), "student_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# -----------------------
# Instructions Section
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
        all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
        dataframes = []
        for sheet_name, sheet_df in all_sheets.items():
            sheet_df.columns = sheet_df.columns.str.strip().str.replace(" ", "_")
            if "Class" not in sheet_df.columns:
                sheet_df["Class"] = sheet_name
            dataframes.append(sheet_df)
        df = pd.concat(dataframes, ignore_index=True)

    df.columns = df.columns.str.strip().str.replace(" ", "_")

    df["CA_Percent"] = pd.to_numeric(df["CA_Percent"], errors="coerce")
    df["Exam_Percent"] = pd.to_numeric(df["Exam_Percent"], errors="coerce")
    df["Overall"] = df["Total"].apply(lambda x: "ABS" if str(x).strip().upper() == "ABS" else pd.to_numeric(x, errors="coerce"))

    df["Grade"] = df["Grade"].str.strip().str.upper()
    grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]
    df["Grade"] = pd.Categorical(df["Grade"], categories=grade_order, ordered=True)

    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    benchmark = numeric_overall.mean()
    df["Above_Benchmark"] = numeric_overall > benchmark

    # -----------------------
    # Display Data
    # -----------------------
    st.subheader("📌 Preview Data")
    st.dataframe(df.head())

    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df[["CA_Percent", "Exam_Percent", "Overall"]].apply(pd.to_numeric, errors='coerce').describe())
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # -----------------------
    # Grade Distribution
    # -----------------------
    st.subheader("🎓 Grade Distribution")
    st.markdown("#### 🧮 Overall")
    grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(grade_dist.rename("Count"))
    with col2:
        st.bar_chart(grade_dist)
    
    # Create version for PDF (with numbers on bars)
    fig_overall_bar, ax = plt.subplots()
    bars = grade_dist.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Number of Students")
    ax.set_xlabel("Grade")

    # Add numbers on top of each bar (PDF only)
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom',
                fontsize=10)

    fig_overall_bar.tight_layout()
    overall_bar_chart_path = os.path.join(tempfile.gettempdir(), "overall_grade_bar_chart.png")
    fig_overall_bar.savefig(overall_bar_chart_path, bbox_inches="tight", dpi=300)
    plt.close(fig_overall_bar)

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
                
    # Generate class distribution charts for PDF with numbers
    class_bar_chart_paths = {}
    for class_name in classes:
        class_df = df[df["Class"] == class_name]
        class_grade_dist = class_df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)

        fig, ax = plt.subplots()
        bars = class_grade_dist.plot(kind="bar", ax=ax, color="lightgreen")
        ax.set_title(f"Grade Distribution - Class {class_name}")
        ax.set_ylabel("Number of Students")
        ax.set_xlabel("Grade")
        
        # Add numbers on bars (PDF only)
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=10)
        
        path = os.path.join(tempfile.gettempdir(), f"class_{class_name}_bar_chart.png")
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        
        class_bar_chart_paths[class_name] = path
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # -----------------------
    # Class Performance Summary
    # -----------------------
    st.subheader("🏫 Class Performance Summary")
    st.dataframe(df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].describe().round(2))

    st.subheader("📌 Overall Mean Score by Class")
    df["Numeric_Overall"] = numeric_overall
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    df.groupby("Class")["Numeric_Overall"].mean().plot(kind="bar", ax=ax1)
    ax1.set_title("Grade Distribution by Class")
    ax1.set_ylabel("Mean Overall")
    ax1.set_xlabel("Class")
    st.pyplot(fig1)
    grade_bar_chart_path = os.path.join(tempfile.gettempdir(), "grade_bar_chart.png")
    fig1.savefig(grade_bar_chart_path, bbox_inches="tight")

    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # -----------------------
    # Benchmark Analysis
    # -----------------------
    st.subheader("📉 Above/Below Average Performance")
    above = df["Above_Benchmark"].sum()
    below = len(df) - above
    col1, col2 = st.columns(2)
    col1.metric("Above Benchmark", above)
    col2.metric("Below Benchmark", below)
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # -----------------------
    # Pie Chart
    # -----------------------
    st.subheader("📉  Overall Grade Distribution (Pie Chart)")
    overall_grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0)
    total_students = overall_grade_dist.sum()
    percentage_labels = [f"{g}: {int(c)} students ({c/total_students:.1%})" for g, c in overall_grade_dist.items() if c > 0]

    fig2, ax2 = plt.subplots()
    ax2.pie(overall_grade_dist[overall_grade_dist > 0], labels=percentage_labels,
            labeldistance=1.1, textprops={'fontsize': 12}, autopct='%1.1f%%', startangle=140, counterclock=False)
    st.pyplot(fig2)
    overall_pie_path = os.path.join(tempfile.gettempdir(), "overall_pie_chart.png")
    fig2.savefig(overall_pie_path, bbox_inches="tight")

    # -----------------------
    # Cluster Analysis
    # -----------------------
    st.subheader("🔍 Cluster Analysis (CA% vs Exam%)")
    valid_scores = df[pd.to_numeric(df["Overall"], errors="coerce").notna()]
    cluster_chart_path = None

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
        ax.legend()
        st.pyplot(fig)
        cluster_chart_path = os.path.join(tempfile.gettempdir(), "cluster_chart.png")
        fig.savefig(cluster_chart_path, bbox_inches="tight")
    else:
        st.info("❗ Not enough valid numeric data to perform clustering.")

    # -----------------------
    # Sidebar Configuration
    # -----------------------
    with st.sidebar:
        st.header("Export Options")
        
        # PDF Report Options
        pdf_title = st.text_input("PDF Report Title", "Student Performance Report")
        pdf_filename = st.text_input("PDF Filename (without .pdf)", "student_performance_report")
        
        # CSV Export Options
        csv_filename = st.text_input("CSV Filename (without .csv)", "student_analysis_results")
        
        st.markdown("---")
        
        # PDF Report Button
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating report..."):
                report_path = generate_pdf_report(
                    df=df,
                    benchmark=benchmark,
                    grade_dist=grade_dist,
                    grade_by_class=df.groupby(['Class', 'Grade']).size().unstack(fill_value=0).reindex(columns=grade_order, fill_value=0),
                    overall_pie_path=overall_pie_path,
                    cluster_chart_path=cluster_chart_path,
                    overall_bar_chart_path=overall_bar_chart_path,
                    class_bar_chart_paths=class_bar_chart_paths,
                    report_title=pdf_title  # Pass the user-defined title
                )
            
            with open(report_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report", 
                    data=f, 
                    file_name=f"{pdf_filename}.pdf", 
                    mime="application/pdf"
                )
        
        # CSV Export Button
        st.download_button(
            "📥 Download Analyzed CSV", 
            data=df.to_csv(index=False), 
            file_name=f"{csv_filename}.csv", 
            mime="text/csv"
        )