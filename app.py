import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tempfile
import os
import base64
import pdfkit
import pathlib

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

    df["CA_Percent"] = (pd.to_numeric(df["CA_Marks"], errors="coerce") / ca_total) * ca_weight
    df["Exam_Percent"] = (pd.to_numeric(df["Exam_Marks"], errors="coerce") / exam_total) * exam_weight

    def compute_overall(ca, exam):
        if str(ca).strip().upper() == "ABS" or str(exam).strip().upper() == "ABS":
            return "ABS"
        try:
            return float(ca) + float(exam)
        except:
            return None

   # Use 'Total' column directly as Overall (respect ABS if present)
    df["Overall"] = df["Total"].apply(lambda x: "ABS" if str(x).strip().upper() == "ABS" else pd.to_numeric(x, errors="coerce"))

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
    # -----------------------
# PDF Report Generation
# -----------------------
if st.button("📄 Generate PDF Report"):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Get current year for report title
            current_year = df['Class'].str.extract(r'(\d+)')[0].unique()[0] if not df.empty else "N/A"
            
            # Ensure the temp directory exists
            os.makedirs(temp_dir, exist_ok=True)
            
            # Prepare all data tables first
            # 1. Descriptive Statistics Table
            stats_df = df[["CA_Percent", "Exam_Percent", "Overall"]].apply(pd.to_numeric, errors='coerce').describe().round(2)
            stats_html = stats_df.to_html(border=1, classes='table table-striped')
            
            # 2. Enhanced Grade Distribution Table by Class
            grade_table = df.groupby(['Class', 'Grade']).size().unstack(fill_value=0)
            grade_table['Total'] = grade_table.sum(axis=1)
            grade_table = grade_table.reindex(columns=grade_order + ['Total'])
            grade_table_html = grade_table.to_html(border=1, classes='table table-striped')
            
            # 3. Create visualizations
            # Grade Distribution Chart (full page)
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            grade_by_class.plot(kind="bar", ax=ax1, width=0.8)
            ax1.set_ylabel("Number of Students", fontsize=12)
            ax1.set_xlabel("Class", fontsize=12)
            ax1.set_title(f"Grade Distribution by  ({current_year})", fontsize=14, pad=20)
            ax1.legend(title="Grade", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
            plt.tight_layout()
            grade_dist_path = os.path.join(temp_dir, "grade_dist.png")
            fig1.savefig(grade_dist_path, bbox_inches='tight', dpi=300)
            plt.close(fig1)
            
            # Pie Chart (full page)
            overall_grade_dist = df["Grade"].value_counts().reindex(grade_order).fillna(0)
            total_students = overall_grade_dist.sum()
            percentage_labels = [f"{g}: {int(c)} students ({c/total_students:.1%})" 
                               for g, c in overall_grade_dist.items() if c > 0]
            
            fig2, ax2 = plt.subplots(figsize=(10, 10))
            ax2.pie(overall_grade_dist[overall_grade_dist > 0],
                    labels=percentage_labels,
                    labeldistance=1.1,
                    textprops={'fontsize': 12},
                    autopct='%1.1f%%',
                    startangle=140,
                    counterclock=False,
                    pctdistance=0.85)
            ax2.set_title(f"Overall Grade Distribution ({current_year})", fontsize=14, pad=20)
            plt.tight_layout()
            pie_chart_path = os.path.join(temp_dir, "pie_chart.png")
            fig2.savefig(pie_chart_path, bbox_inches='tight', dpi=300)
            plt.close(fig2)
            
            # Cluster Plot (full page)
            cluster_path = ""
            valid_scores = df[pd.to_numeric(df["Overall"], errors="coerce").notna()]
            if not valid_scores.empty:
                kmeans = KMeans(n_clusters=3, n_init=10)
                valid_scores = valid_scores.copy()
                valid_scores["Cluster"] = kmeans.fit_predict(valid_scores[["CA_Percent", "Exam_Percent"]])
                
                fig3, ax3 = plt.subplots(figsize=(10, 8))
                for cluster in sorted(valid_scores["Cluster"].unique()):
                    cluster_df = valid_scores[valid_scores["Cluster"] == cluster]
                    ax3.scatter(cluster_df["CA_Percent"], cluster_df["Exam_Percent"], 
                               label=f"Cluster {int(cluster)}", s=100)
                ax3.set_xlabel("CA Percentage", fontsize=12)
                ax3.set_ylabel("Exam Percentage", fontsize=12)
                ax3.set_title(f"Student Performance Clusters ({current_year})", fontsize=14, pad=20)
                ax3.legend(fontsize=10)
                ax3.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                cluster_path = os.path.join(temp_dir, "cluster.png")
                fig3.savefig(cluster_path, bbox_inches='tight', dpi=300)
                plt.close(fig3)
            
            # HTML Content with improved styling
            html_content = f"""
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{ 
                        font-family: Arial, sans-serif; 
                        padding: 20px;
                        line-height: 1.6;
                    }}
                    .report-title {{
                        color: #2e6c80;
                        text-align: center;
                        margin-bottom: 30px;
                        border-bottom: 2px solid #2e6c80;
                        padding-bottom: 10px;
                    }}
                    .section-title {{
                        color: #2e6c80;
                        margin-top: 30px;
                        margin-bottom: 15px;
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 5px;
                    }}
                    .table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin-top: 20px;
                        margin-bottom: 30px;
                    }}
                    .table th, .table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: center;
                    }}
                    .table th {{
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }}
                    .chart-container {{
                        page-break-after: always;
                        margin-top: 20px;
                        margin-bottom: 40px;
                    }}
                    .chart-title {{
                        text-align: center;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }}
                    .chart-img {{
                        width: 100%;
                        max-width: 800px;
                        display: block;
                        margin: 0 auto;
                    }}
                    .class-table {{
                        margin-bottom: 40px;
                    }}
                </style>
            </head>
            <body>
                <h1 class="report-title">📊 Year {current_year} Student Performance Report </h1>
                
                <div class="chart-container">
                    <h2 class="section-title">📈 Descriptive Statistics</h2>
                    {stats_html}
                    <br>
                    <br>
                    <h2 class="section-title">🎓 Grade Distribution</h2>
                    <h3>By Class with Totals</h3>
                    {grade_table_html}
                </div>
                
                <div class="chart-container">
                    <h2 class="section-title">🎯 Grade Distribution by Class</h2>
                    <div class="chart-title">Visual representation of grade distribution across classes</div>
                    <img class="chart-img" src="{pathlib.Path(grade_dist_path).as_uri()}" alt="Grade Distribution Bar Chart">
                </div>
                
                <div class="chart-container">
                    <h2 class="section-title">📌 Overall Grade Distribution</h2>
                    <div class="chart-title">Percentage distribution of all grades across all classes</div>
                    <img class="chart-img" src="{pathlib.Path(pie_chart_path).as_uri()}" alt="Grade Distribution Pie Chart">
                </div>
            """
            
            if cluster_path:
                html_content += f"""
                <div class="chart-container">
                    <h2 class="section-title">🔍 Cluster Analysis</h2>
                    <div class="chart-title">Student performance clusters based on CA vs Exam percentages</div>
                    <img class="chart-img" src="{pathlib.Path(cluster_path).as_uri()}" alt="Cluster Scatter Plot">
                </div>
                """

            # PDF Generation with better page breaks
            options = {
                'enable-local-file-access': None,
                'quiet': '',
                'margin-top': '15mm',
                'margin-right': '15mm',
                'margin-bottom': '15mm',
                'margin-left': '15mm',
                'page-size': 'A4',
                'orientation': 'Portrait',
                'encoding': 'UTF-8',
            }
            
            pdf_path = os.path.join(temp_dir, "report.pdf")
            pdfkit.from_string(html_content, pdf_path, options=options)
            
            # Provide download link
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"student_performance_report_{current_year}.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.error(f"Error generating PDF report: {str(e)}")
 
