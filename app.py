# =======================
# Imports & Config
# =======================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import numpy as np
from PIL import Image
from pandas.plotting import table
import gspread
from google.oauth2.service_account import Credentials
import re

# Set matplotlib to not use interactive backend to prevent memory issues
plt.switch_backend('Agg')

st.set_page_config(page_title="Student Performance Dashboard", layout="wide")
st.title("📊 Student Performance Dashboard")

# =======================
# PDF Generation Function
# =======================
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

    # Report Title
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
    
    # Add gender distribution chart
    if gender_bar_chart_path and os.path.exists(gender_bar_chart_path):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Gender Distribution by Class", ln=2)
        pdf.ln(5)
        pdf.image(gender_bar_chart_path, w=180)
    
    # ----------------- Class-wise Grade Distributions -----------------
    for class_name in class_bar_chart_paths.keys():
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Class: {class_name}", ln=True)
        pdf.ln(4)

        page_width = pdf.w - 2*pdf.l_margin
        chart_width = page_width
        chart_height = 120
        
        # Add bar chart
        if os.path.exists(class_bar_chart_paths[class_name]):
            pdf.image(class_bar_chart_paths[class_name], 
                      x=pdf.l_margin, w=chart_width, h=chart_height)
        pdf.ln(10)
        
        # Add pie chart
        if class_name in class_pie_chart_paths and os.path.exists(class_pie_chart_paths[class_name]):
            pdf.image(class_pie_chart_paths[class_name], 
                      x=pdf.l_margin, w=chart_width, h=chart_height)
        pdf.ln(10)

    # Save PDF
    pdf_path = os.path.join(tempfile.gettempdir(), "student_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# =======================
# Google Sheets Functions
# =======================
def authenticate_google_sheets():
    """Authenticate with Google Sheets API using service account credentials"""
    try:
        # Create the credentials object
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        
        # Check if the JSON file exists
        creds_file = "google_credentials.json"
        if not os.path.exists(creds_file):
            st.error("Google credentials file not found. Please ensure 'google_credentials.json' is in the same directory.")
            return None
            
        credentials = Credentials.from_service_account_file(creds_file, scopes=scopes)
        gc = gspread.authorize(credentials)
        return gc
    except Exception as e:
        st.error(f"Failed to authenticate with Google Sheets: {str(e)}")
        return None

def extract_sheet_id_from_url(url):
    """Extract the Google Sheet ID from a URL"""
    # Pattern to match Google Sheet URLs
    patterns = [
        r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)',
        r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)/edit',
        r'https://docs\.google\.com/spreadsheets/d/([a-zA-Z0-9-_]+)/.*'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None

def load_google_sheet(url):
    """Load data from a Google Sheet - always loads all worksheets"""
    try:
        gc = authenticate_google_sheets()
        if not gc:
            return None
            
        sheet_id = extract_sheet_id_from_url(url)
        if not sheet_id:
            st.error("Invalid Google Sheets URL. Please provide a valid URL.")
            return None
            
        # Open the spreadsheet
        spreadsheet = gc.open_by_key(sheet_id)
        
        # Get all worksheets and combine them
        worksheets = spreadsheet.worksheets()
        dataframes = []
        
        for worksheet in worksheets:
            df = pd.DataFrame(worksheet.get_all_records())
            if "Class" not in df.columns:
                df["Class"] = worksheet.title
            dataframes.append(df)
        
        return pd.concat(dataframes, ignore_index=True)
            
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")
        return None

# =======================
# Instructions Section
# =======================
with st.expander("📘 Instructions: Preparing Your Excel File", expanded=True):
    st.markdown("""
    Please ensure your Excel or CSV file includes the following columns:

    - **Name**: Student's full name  
    - **Gender**: e.g., M(Male) / F (Female)
    - **Class**: Class name or code
    - **Assessment**: Continuous Assessment percentage (%, numeric)
    - **Examination**: Examination percentage (%, numeric)
    - **Total**: Total marks between Assessment and Examination (%, numeric)
    - **Grade**: Student's grade (A*, A, B, C, D, E, U, ABS)  

    **Example Column Headers**:
    ```
    Name, Gender, Class, Assessment, Examination, Total, Grade
    ```

    **Note**:
    - The column names are **case-insensitive**, but ensure they follow the exact format.
    - Avoid extra spaces in headers or leave cells empty.
    - 'ABS' will be treated as **Absent** and excluded from numerical computations.
    """)

# ================================
# File Upload & Preprocessing
# ================================
input_method = st.radio(
    "Select input method:",
    ["Upload File", "Google Sheets URL"],
    horizontal=True
)

df = None

if input_method == "Upload File":
    uploaded_file = st.file_uploader("📁 Upload the student CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file:
        # --- File Reading ---
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
        
else:  # Google Sheets URL
    sheets_url = st.text_input("🔗 Google Sheets URL", placeholder="https://docs.google.com/spreadsheets/d/...")
    
    if sheets_url:
        with st.spinner("Loading data from Google Sheets..."):
            df = load_google_sheet(sheets_url)

# Only process data if df is not None
if df is not None and not df.empty:
    # Process the data
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    
    # --- Flexible Column Matching ---
    def find_column(df, keywords):
        for col in df.columns:
            if any(keyword in col.lower() for keyword in keywords):
                return col
        return None
    
    # The rest of your data processing code...
    ca_keywords = ["ca_percent", "assessment", "continuous_assessment", "ca_score"]
    exam_keywords = ["exam_percent", "examination", "final_exam", "exam_score"]
    gender_keywords = ["gender", "sex", "male/female", "m/f"]

    assessment_col = find_column(df, ca_keywords)
    examination_col = find_column(df, exam_keywords)
    gender_col = find_column(df, gender_keywords)

    # --- Error Handling for Missing Columns ---
    if assessment_col is None or examination_col is None:
        error_msg = "Error: Required columns not found.\n\nLooking for:"
        if assessment_col is None:
            error_msg += f"\n- CA/Assessment column (tried: {', '.join(ca_keywords)})"
        if examination_col is None:
            error_msg += f"\n- Exam column (tried: {', '.join(exam_keywords)})"
        error_msg += f"\n\nAvailable columns: {', '.join(df.columns)}"
        st.error(error_msg)
        st.stop()

    # --- Gender Processing ---
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

    # --- Rename Columns ---
    df = df.rename(columns={assessment_col: "CA_Percent", examination_col: "Exam_Percent"})
    df["CA_Percent"] = pd.to_numeric(df["CA_Percent"], errors="coerce")
    df["Exam_Percent"] = pd.to_numeric(df["Exam_Percent"], errors="coerce")

    # --- Overall Score ---
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

    # --- Grade Column ---
    grade_keywords = ["grade", "letter_grade", "final_grade"]
    grade_col = find_column(df, grade_keywords)
    if grade_col:
        df["Grade"] = df[grade_col].str.strip().str.upper()
        grade_order = ["A*", "A", "B", "C", "D", "E", "U", "ABS"]
        df["Grade"] = pd.Categorical(df["Grade"], categories=grade_order, ordered=True)
    else:
        df["Grade"] = np.nan
        st.warning("No grade column found - you may want to calculate grades")

    # --- Benchmark ---
    numeric_overall = pd.to_numeric(df["Overall"], errors="coerce")
    benchmark = numeric_overall.mean()
    st.success(f"Data loaded successfully! Found {len(df)} records.")

    # ================================
    # Gender Distribution by Class
    # ================================
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
            
            # Define pastel colors
            pastel_blue = 'background-color: #ADD8E6'  # Light blue for male higher
            pastel_pink = 'background-color: #FFB6C1'  # Light pink for female higher
            white_bg = 'background-color: white'       # White for lower value
            pastel_lavender = 'background-color: #E6E6FA'  # Lavender for equal values
            
            # Create a style DataFrame
            def color_comparison(val, other_val, higher_style, lower_style):
                if val > other_val:
                    return higher_style
                elif val < other_val:
                    return lower_style
                else:
                    return pastel_lavender
            
            # Initialize style DataFrame with white background
            style_df = pd.DataFrame(white_bg, index=gender_table.index, columns=gender_table.columns)
            
            # Apply colors to count columns (M and F)
            for idx in gender_table.index:
                m_val = gender_table.loc[idx, 'M']
                f_val = gender_table.loc[idx, 'F']
                
                style_df.loc[idx, 'M'] = color_comparison(m_val, f_val, pastel_blue, white_bg)
                style_df.loc[idx, 'F'] = color_comparison(f_val, m_val, pastel_pink, white_bg)
                
                # Apply colors to percentage columns (% Male and % Female)
                m_pct = gender_table.loc[idx, '% Male']
                f_pct = gender_table.loc[idx, '% Female']
                
                style_df.loc[idx, '% Male'] = color_comparison(m_pct, f_pct, pastel_blue, white_bg)
                style_df.loc[idx, '% Female'] = color_comparison(f_pct, m_pct, pastel_pink, white_bg)
            
            # Create styled table
            styled_table = gender_table.style \
                .format({'% Male': '{:.1f}%', '% Female': '{:.1f}%'}) \
                .apply(lambda x: style_df, axis=None)
            
            st.dataframe(styled_table, use_container_width=True)
            
            # Save table as image for PDF
            fig_table, ax_table = plt.subplots(figsize=(10, 4))
            ax_table.axis('off')
            
            # For matplotlib table, apply similar coloring
            cell_colors = []
            for _, row in gender_table.iterrows():
                row_colors = []
                for col in gender_table.columns:
                    if col == 'M':
                        if row['M'] > row['F']:
                            row_colors.append(pastel_blue.split(': ')[1])
                        elif row['M'] < row['F']:
                            row_colors.append('white')
                        else:
                            row_colors.append(pastel_lavender.split(': ')[1])
                    elif col == 'F':
                        if row['F'] > row['M']:
                            row_colors.append(pastel_pink.split(': ')[1])
                        elif row['F'] < row['M']:
                            row_colors.append('white')
                        else:
                            row_colors.append(pastel_lavender.split(': ')[1])
                    elif col == '% Male':
                        if row['% Male'] > row['% Female']:
                            row_colors.append(pastel_blue.split(': ')[1])
                        elif row['% Male'] < row['% Female']:
                            row_colors.append('white')
                        else:
                            row_colors.append(pastel_lavender.split(': ')[1])
                    elif col == '% Female':
                        if row['% Female'] > row['% Male']:
                            row_colors.append(pastel_pink.split(': ')[1])
                        elif row['% Female'] < row['% Male']:
                            row_colors.append('white')
                        else:
                            row_colors.append(pastel_lavender.split(': ')[1])
                    else:
                        row_colors.append('white')
                cell_colors.append(row_colors)
            
            table(ax_table, gender_table.round(1), loc='center', cellLoc='center', cellColours=cell_colors)
            gender_table_path = os.path.join(tempfile.gettempdir(), "gender_table.png")
            fig_table.savefig(gender_table_path, bbox_inches='tight', dpi=200)
            plt.close(fig_table)
    else:
        gender_bar_chart_path = None
        gender_table_path = None
        
    st.markdown("<hr style='border:1px solid lightgrey'>", unsafe_allow_html=True)


    # =======================
    # Grade Distribution
    # =======================
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
            # Adjust column ratios to better accommodate the wider table
            col1, col2 = st.columns([1.3, 2])  # Increased first column width
            
            with col1:
                # Create DataFrame
                display_df = pd.DataFrame({
                    'Grade': class_grade_dist.index,
                    'Count': class_grade_dist.values,
                    '%': (class_grade_dist.values / total_class_students * 100).round(1)
                }).set_index('Grade')

                # Get unique sorted percentages to find 2nd lowest
                sorted_percentages = np.sort(display_df['%'].unique())
                
                # Define masks for min, 2nd min, and max
                min_mask = display_df['%'] == sorted_percentages[0]  # Lowest
                second_min_mask = display_df['%'] == sorted_percentages[1] if len(sorted_percentages) > 1 else pd.Series(False, index=display_df.index)  # 2nd lowest
                max_mask = display_df['%'] == sorted_percentages[-1]  # Highest

                # Apply styling
                styled_df = (
                    display_df.style
                    .format({'%': '{:.1f}%'})
                    # Alternate row colors (light gray/white)
                    .set_properties(**{'background-color': '#f9f9f9'}, subset=pd.IndexSlice[::2, :])
                    # Highlight min (pastel red)
                    .set_properties(**{'background-color': '#ffcccc'}, subset=pd.IndexSlice[min_mask, :])
                    # Highlight 2nd min (pastel yellow)
                    .set_properties(**{'background-color': '#fff2cc'}, subset=pd.IndexSlice[second_min_mask, :])
                    # Highlight max (pastel green)
                    .set_properties(**{'background-color': '#ccffcc'}, subset=pd.IndexSlice[max_mask, :])
                )

                st.table(styled_df)

            
            with col2:
                # Rest of your chart code remains the same
                tab1, tab2 = st.tabs(["📊 Bar Chart", "🥧 Pie Chart"])
                
                with tab1:
                    st.bar_chart(class_grade_dist)
                
                with tab2:
                    if total_class_students > 0:
                        fig_pie, ax_pie = plt.subplots(figsize=(3, 2.2))
                        pie_data = class_grade_dist[class_grade_dist > 0]
                        pie_labels = [f"{g} ({c})" for g, c in zip(pie_data.index, pie_data)]
                        ax_pie.pie(pie_data, 
                                labels=pie_labels,
                                autopct='%1.1f%%',
                                startangle=140,
                                textprops={'fontsize': 8})
                        ax_pie.set_title(f"Grade Distribution - {class_name}", fontsize=10)
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

    # =========================
    # Class Performance Summary
    # =========================
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

    # =======================
    # Overall Pie Chart
    # =======================
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
else:
    # Show a message if no data is loaded yet
    st.info("Please upload a file or provide a Google Sheets URL to begin analysis.")   
     
# =========================
# Sidebar (Export Options)
# =========================
with st.sidebar:
    st.header("Export Options")

    # --- Inputs (always available) ---
    pdf_title = st.text_input("PDF Report Title", "Student Performance Report")
    pdf_filename = st.text_input("PDF Filename (without .pdf)", "student_performance_report")
    csv_filename = st.text_input("CSV Filename (without .csv)", "student_analysis_results")

    st.markdown("---")

    # --- Export Buttons ---
    # Check if df exists and is not None
    df_exists = "df" in locals() and df is not None
    
    if df_exists:
        # Enable buttons once data is loaded
        pdf_button = st.button("📄 Generate PDF Report", use_container_width=True)
        if pdf_button:
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
                    mime="application/pdf",
                    use_container_width=True
                )

        # CSV download button
        csv_data = df.to_csv(index=False) if df_exists else ""
        st.download_button(
            "📥 Download Analyzed CSV", 
            data=csv_data, 
            file_name=f"{csv_filename}.csv", 
            mime="text/csv",
            use_container_width=True,
            disabled=not df_exists
        )
    else:
        # Greyed-out style for disabled buttons
        st.markdown(
            """
            <style>
            .stButton>button[disabled], .stDownloadButton>button[disabled] {
                background-color: #e0e0e0 !important;
                color: #888 !important;
                cursor: not-allowed !important;
                border: 1px solid #ccc !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.button("📄 Generate PDF Report", disabled=True, use_container_width=True)
        st.download_button(
            "📥 Download Analyzed CSV", 
            data="", 
            file_name=f"{csv_filename}.csv", 
            mime="text/csv",
            disabled=True,
            use_container_width=True
        )
        st.info("📂 Upload a file or provide a Google Sheets URL to enable exports.")