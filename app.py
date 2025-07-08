import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fpdf import FPDF
import tempfile
import os
import numpy as np
from PIL import Image
from pandas.plotting import table

# Set matplotlib to not use interactive backend to prevent memory issues
plt.switch_backend('Agg')

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# -----------------------
# PDF Generation Function
# -----------------------
def generate_pdf_report(df, benchmark, grade_dist, grade_by_class, overall_pie_path, 
                       overall_bar_chart_path, class_bar_chart_paths, class_pie_chart_paths, 
                       class_grade_bar_path, class_grade_table_path,
                       gender_bar_chart_graph, gender_table_path=None, report_title="Student Performance Report"):
    
    class PDFWithFooter(FPDF):
        def footer(self):
            self.set_y(-15)  # Position at 1.5 cm from bottom
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    pdf = PDFWithFooter()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.alias_nb_pages()  # Enable page numbering

    # Now add the title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, report_title, ln=True, align='C')

    # ----------------- Overall Grade Distribution Charts -----------------
    pdf.set_font("Arial", 'B', 14)
    pdf.ln(3)
    pdf.cell(0, 10, "Class-wise Grade Distribution", ln=2)
    
    # Add the bar chart
    if os.path.exists(class_grade_bar_path):
        pdf.set_font("Arial", 'I', 12)
        pdf.cell(0, 8, "Visualization: Number of Students by Grade and Class", ln=1)
        pdf.image(class_grade_bar_path, x=10, w=190)
        pdf.ln(5)
    
    # Add the data table
    if gender_bar_chart_path and os.path.exists(gender_bar_chart_path):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Gender Distribution by Class", ln=2)
        pdf.ln(5)
        pdf.image(gender_bar_chart_path, w=180)
    
    # ----------------- Class-wise Grade Distributions -----------------
    for class_name in class_bar_chart_paths.keys():
        # Start a new page for each class
        pdf.add_page()
        
        # Class title
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Class: {class_name}", ln=True)
        pdf.ln(4)  # Small space after title
        
        # Calculate positions - we'll place charts vertically with full width
        page_width = pdf.w - 2*pdf.l_margin  # Available width
        chart_width = page_width  # Full width for each chart
        chart_height = 120  # Height in mm (adjust as needed)
        
        # Add bar chart (top)
        if os.path.exists(class_bar_chart_paths[class_name]):
            pdf.image(class_bar_chart_paths[class_name], 
                    x=pdf.l_margin, 
                    y=None,  # Let FPDF determine Y position
                    w=chart_width,
                    h=chart_height)
        
        # Add some space between charts
        pdf.ln(10)
        
        # Add pie chart (bottom) if it exists
        if class_name in class_pie_chart_paths and os.path.exists(class_pie_chart_paths[class_name]):
            pdf.image(class_pie_chart_paths[class_name], 
                    x=pdf.l_margin, 
                    y=None,  # Let FPDF determine Y position
                    w=chart_width,
                    h=chart_height)
        
        # Add some space after the charts
        pdf.ln(10)

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
    - **Gender**: e.g., M(Male) / F (Female)
    - **Class**: Class name or code
    - **Assessment**: Continuous Assessment percentage (%, numeric)
    - **Examination**: Examination percentage (%, numeric)
    - **Total**: Total marks between Assessment and Examination (%, numeric)
    - **Grade**: Student's grade (A*, A, B, C, D, E.......)  

    **Example Column Headers**:
    ```
    Name, Gender, Class, Assessment, Examination, Total, Grade
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
    # Read the file
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

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    
    # Define flexible column matching
    ca_keywords = ["assessment", "ca", "ca_percent", "continuous_assessment", "ca_score"]
    exam_keywords = ["exam", "examination", "exam_percent", "final_exam", "exam_score"]
    gender_keywords = ["gender", "sex", "male/female", "m/f"]
   
    
    # Find relevant columns
    def find_column(df, keywords):
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None
    
    assessment_col = find_column(df, ca_keywords)
    examination_col = find_column(df, exam_keywords)
    gender_col = find_column(df, gender_keywords)

    # Handle missing columns with error message
    if assessment_col is None or examination_col is None:
        error_msg = "Error: Required columns not found.\n\nLooking for:"
        if assessment_col is None:
            error_msg += f"\n- CA/Assessment column (tried: {', '.join(ca_keywords)})"
        if examination_col is None:
            error_msg += f"\n- Exam column (tried: {', '.join(exam_keywords)})"
        error_msg += f"\n\nAvailable columns: {', '.join(df.columns)}"
        st.error(error_msg)
        st.stop()
    
    if gender_col:
        df = df.rename(columns={gender_col: "Gender"})
        # Convert to string and clean
        df['Gender'] = df['Gender'].astype(str).str.strip().str.upper()
        
        # Map common variations to M/F
        gender_mapping = {
            'M': 'M', 'MALE': 'M', 'BOY': 'M',
            'F': 'F', 'FEMALE': 'F', 'GIRL': 'F'
        }
        df['Gender'] = df['Gender'].map(gender_mapping).fillna(df['Gender'])
        
        # Take first character if not already M/F (e.g., "Male" → "M")
        df['Gender'] = df['Gender'].str[0]
        df['Gender'] = df['Gender'].where(df['Gender'].isin(['M', 'F']), None)
        
        
        if not df['Gender'].isin(['M', 'F']).any():
            st.warning("⚠️ Gender column found but contains no valid M/F data - gender analysis will be skipped")
            gender_col = None
    else:
        st.warning("⚠️ Gender column not found - gender analysis will be skipped. Expected column names: 'Gender', 'Sex', 'M/F'")
        gender_col = None

    # Rename columns for consistency
    df = df.rename(columns={
        assessment_col: "CA_Percent",
        examination_col: "Exam_Percent"
    })
    
    # Convert to numeric with error handling
    df["CA_Percent"] = pd.to_numeric(df["CA_Percent"], errors="coerce")
    df["Exam_Percent"] = pd.to_numeric(df["Exam_Percent"], errors="coerce")
    
    # Process Overall column
    total_keywords = ["total", "overall", "final_score", "combined"]
    total_col = find_column(df, total_keywords)
    
    if total_col:
        df["Overall"] = df[total_col].apply(
            lambda x: "ABS" if str(x).strip().upper() == "ABS" 
            else pd.to_numeric(x, errors="coerce")
        )
    else:
        # Calculate overall if no total column exists
        df["Overall"] = df["CA_Percent"] + df["Exam_Percent"]
        st.warning("No total/overall column found - calculated sum of CA and Exam scores")

    # Process Grade column if it exists
    grade_keywords = ["grade", "letter_grade", "final_grade"]
    grade_col = find_column(df, grade_keywords)
    
    if grade_col:
        df["Grade"] = df[grade_col].str.strip().str.upper()
        grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]
        df["Grade"] = pd.Categorical(df["Grade"], categories=grade_order, ordered=True)
    else:
        df["Grade"] = np.nan
        st.warning("No grade column found - you may want to calculate grades")

    # Calculate benchmark
    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    benchmark = numeric_overall.mean()

    # Show success message
    st.success(f"Data loaded successfully! Found {len(df)} records.")
    
    # -----------------------
    # Gender Distribution by Class (Tabbed View)
    # -----------------------
    if 'Gender' in df.columns and df['Gender'].isin(['M', 'F']).any():
        st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
        st.subheader("👥 Gender Distribution by Class")
        
        # Get gender counts per class
        gender_by_class = df.groupby(['Class', 'Gender']).size().unstack(fill_value=0)
        gender_by_class = gender_by_class.reindex(columns=['M', 'F'], fill_value=0)
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Bar Chart", "📋 Data Table"])
        
        with tab1:
            # Grouped bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar positions and width
            classes = gender_by_class.index
            x = np.arange(len(classes))
            bar_width = 0.35
            
            # Plot bars
            bars_m = ax.bar(x - bar_width/2, gender_by_class['M'], bar_width, 
                        label='Male', color='lightblue')
            bars_f = ax.bar(x + bar_width/2, gender_by_class['F'], bar_width, 
                        label='Female', color='pink')
            
            # Customize chart
            ax.set_xlabel('Class')
            ax.set_ylabel('Number of Students')
            ax.set_title('Gender Distribution by Class')
            ax.set_xticks(x)
            ax.set_xticklabels(classes, rotation=45)
            ax.legend()
            
            # Add value labels
            for bars in [bars_m, bars_f]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Save for PDF
            gender_bar_chart_path = os.path.join(tempfile.gettempdir(), "gender_by_class_chart.png")
            fig.savefig(gender_bar_chart_path, bbox_inches='tight', dpi=200)
            plt.close(fig)
        
        with tab2:
            # Data table with additional statistics
            st.write("### Detailed Gender Counts by Class")
            
            # Calculate percentages
            gender_table = gender_by_class.copy()
            gender_table['Total'] = gender_table['M'] + gender_table['F']
            gender_table['% Male'] = (gender_table['M'] / gender_table['Total'] * 100).round(1)
            gender_table['% Female'] = (gender_table['F'] / gender_table['Total'] * 100).round(1)
            
            # Format table
            styled_table = gender_table.style \
                .format({'% Male': '{:.1f}%', '% Female': '{:.1f}%'}) \
                .background_gradient(subset=['M', 'F'], cmap='Blues') \
                .background_gradient(subset=['% Male', '% Female'], cmap='YlOrRd')
            
            st.dataframe(styled_table, use_container_width=True)
            
            # Save table as image for PDF
            fig_table, ax_table = plt.subplots(figsize=(10, 4))
            ax_table.axis('off')
            table(ax_table, gender_table.round(1), loc='center', cellLoc='center')
            gender_table_path = os.path.join(tempfile.gettempdir(), "gender_table.png")
            fig_table.savefig(gender_table_path, bbox_inches='tight', dpi=200)
            plt.close(fig_table)
    else:
        gender_bar_chart_path = None
        gender_table_path = None
        
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)
    
    # -----------------------
    # Grade Distribution
    # -----------------------
    st.subheader("🎓 Grade Distribution")
    st.markdown("#### 🧮 Overall")
    
    grade_by_class = df.groupby(['Class', 'Grade']).size().unstack(fill_value=0)
    grade_by_class = grade_by_class.reindex(columns=grade_order, fill_value=0)

    # Create tabs for different views
    tab1, tab2 = st.tabs(["📈 Bar Chart", "📋 Data Table"])

    with tab1:
        # Create the grouped bar chart using matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Set width and positions for the bars
        n_classes = len(grade_by_class)
        n_grades = len(grade_order)
        bar_width = 0.8 / n_grades
        index = np.arange(n_classes)
        
        # Create bars for each grade
        for i, grade in enumerate(grade_order):
            ax.bar(index + i * bar_width, 
                grade_by_class[grade], 
                width=bar_width,
                label=grade,
                color=plt.cm.tab20(i))  # Using a colormap for distinct colors
        
        # Customize the chart
        ax.set_title('Number of Students by Grade and Class')
        ax.set_ylabel('Number of Students')
        ax.set_xlabel('Class')
        ax.set_xticks(index + bar_width * (n_grades / 2 - 0.5))
        ax.set_xticklabels(grade_by_class.index, rotation=45)
        ax.legend(title='Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        # Display the raw data table
        st.dataframe(grade_by_class.style.background_gradient(cmap='Blues'), 
                    use_container_width=True)

    # Save this chart for PDF report
    class_grade_bar_path = os.path.join(tempfile.gettempdir(), "class_grade_bar_chart.png")
    fig.savefig(class_grade_bar_path, bbox_inches='tight', dpi=300)

    fig_table, ax_table = plt.subplots(figsize=(12, 4))
    ax_table.axis('off')
    table(ax_table, grade_by_class, loc='center', cellLoc='center')
    class_grade_table_path = os.path.join(tempfile.gettempdir(), "class_grade_table.png")
    fig_table.savefig(class_grade_table_path, bbox_inches='tight', dpi=300)
    plt.close(fig_table)
    
    st.markdown("#### 📊 Total by Grade")
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

    overall_bar_chart_path = os.path.join(tempfile.gettempdir(), "overall_grade_bar_chart.png")
    fig_overall_bar.savefig(overall_bar_chart_path, bbox_inches="tight", dpi=200)
    plt.close(fig_overall_bar)

    st.markdown("#### 🏷️ Grade By Class")
    classes = sorted(df["Class"].dropna().unique())
    
    # Initialize paths for PDF generation
    class_bar_chart_paths = {}
    class_pie_chart_paths = {}
    
    for class_name in classes:
        class_df = df[df["Class"] == class_name]
        class_grade_dist = class_df["Grade"].value_counts().reindex(grade_order).fillna(0).astype(int)
        total_class_students = class_grade_dist.sum()
        
        with st.expander(f"📘 Class: {class_name}", expanded=False):
            # Create two columns - one for data, one for charts
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(class_grade_dist.rename("Count"))
            
            with col2:
                # Create tabs for different chart types
                tab1, tab2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                
                with tab1:
                    st.bar_chart(class_grade_dist)
                
                with tab2:
                    if total_class_students > 0:
                        fig_pie, ax_pie = plt.subplots(figsize=(4, 3))  # Match bar chart size
                        
                        pie_data = class_grade_dist[class_grade_dist > 0]
                        pie_labels = [f"{g} ({c})" for g, c in zip(pie_data.index, pie_data)]
                        
                        # Adjust these parameters to fit well in the same size figure
                        ax_pie.pie(pie_data, 
                                labels=pie_labels,
                                autopct='%1.1f%%',
                                startangle=140,
                                textprops={'fontsize': 8})  # Adjust label positioning
                        
                        # Make title match bar chart style
                        ax_pie.set_title(f"Grade Distribution - {class_name}", fontsize=10)
                        
                        # Use tight_layout to optimize space
                        plt.tight_layout()
                        
                        st.pyplot(fig_pie)
                        plt.close(fig_pie)
                    else:
                        st.warning("No grade data available for this class")
        
        # Create bar chart for PDF
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        bars = class_grade_dist.plot(kind="bar", ax=ax_bar, color="lightgreen")
        ax_bar.set_title(f"Grade Distribution - Class {class_name}")
        ax_bar.set_ylabel("Number of Students")
        ax_bar.set_xlabel("Grade")
        
        # Add numbers on bars (PDF only)
        for bar in bars.patches:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom',
                    fontsize=10)
        
        bar_path = os.path.join(tempfile.gettempdir(), f"class_{class_name}_bar_chart.png")
        fig_bar.savefig(bar_path, bbox_inches="tight", dpi=200)
        plt.close(fig_bar)
        class_bar_chart_paths[class_name] = bar_path
        
        # Create pie chart for PDF if there are students
        if total_class_students > 0:
            fig_pie_pdf, ax_pie_pdf = plt.subplots(figsize=(6, 4))
            percentage_labels = [f"{g}: {int(c)} students ({c/total_class_students:.1%})" 
                            for g, c in class_grade_dist.items() if c > 0]
            
            ax_pie_pdf.pie(class_grade_dist[class_grade_dist > 0], 
                        labels=percentage_labels,
                        labeldistance=1.1, 
                        textprops={'fontsize': 8},
                        autopct='%1.1f%%', 
                        startangle=140, 
                        counterclock=False)
            ax_pie_pdf.set_title(f"Class {class_name} Grade Distribution")
            
            pie_path = os.path.join(tempfile.gettempdir(), f"class_{class_name}_pie_chart.png")
            fig_pie_pdf.savefig(pie_path, bbox_inches="tight", dpi=200)
            plt.close(fig_pie_pdf)
            class_pie_chart_paths[class_name] = pie_path
            
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)

    # --------------------------
    # Class Performance Summary
    # --------------------------
    st.subheader("🏫 Class Performance Summary")
    st.markdown("""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin-bottom:15px;">
        <medium>
        📊 This table shows statistical summaries (count, mean, std, min, 25%, 50%, 75%, max) for:
        <ul>
            <li>CA_Percent: Continuous Assessment scores (%)</li>
            <li>Exam_Percent: Examination scores (%)</li>
            <li>Overall: Combined total scores between CA and Exam</li>
        </ul>
        Grouped by class for easy comparison.
        </medium>
        </div>
        """, unsafe_allow_html=True)
    
    st.dataframe(df.groupby("Class")[["CA_Percent", "Exam_Percent", "Overall"]].describe().round(2))

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
    plt.close(fig2) 
    
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
                    grade_by_class=grade_by_class,
                    overall_pie_path=overall_pie_path,
                    overall_bar_chart_path=overall_bar_chart_path,
                    class_bar_chart_paths=class_bar_chart_paths,
                    class_pie_chart_paths=class_pie_chart_paths,
                    class_grade_bar_path=class_grade_bar_path,
                    class_grade_table_path=class_grade_table_path,
                    gender_bar_chart_graph=gender_bar_chart_path,
                    gender_table_path=gender_table_path,
                    report_title=pdf_title
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