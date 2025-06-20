# 📊 Student Performance Dashboard

## Overview
A Streamlit-based web application for analyzing and visualizing student performance data. The dashboard processes uploaded Excel/CSV files containing student records and provides comprehensive analytics including grade distributions, class comparisons, and cluster analysis of assessment patterns.

## Features

### 📌 Core Functionality
- **File Processing**: Handles both Excel (multi-sheet) and CSV files
- **Data Analysis**:
  - Grade distribution statistics (overall and by class)
  - Mean score comparisons across classes
  - Benchmark analysis against cohort average
  - K-means clustering of assessment patterns
- **Visualizations**:
  - Interactive bar charts and pie charts
  - Scatter plots for cluster analysis
- **Reporting**:
  - CSV export of analyzed data
  - PDF report generation with all visualizations

### 📂 Data Requirements
Input files should contain these columns (case-insensitive):
- `Name` - Student names
- `Gender` - Male/Female/Other
- `Class` - Class identifier
- `CA_Marks` - Continuous assessment marks (numeric or 'ABS')
- `CA_Percent` - CA percentage
- `Exam_Marks` - Final exam marks (numeric or 'ABS')  
- `Exam_Percent` - Exam percentage
- `Total` - Combined score
- `Grade` - Letter grade (A*, A, B, etc.)

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd student-performance-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually install:
   ```bash
   pip install streamlit pandas matplotlib scikit-learn fpdf
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. In your browser:
   - Upload a properly formatted Excel/CSV file
   - Explore the interactive visualizations
   - Generate and download reports

## Report Contents
Generated PDF reports include:
    - Summary statistics
    - Grade distribution tables
    - Class performance comparisons  
    - All visualizations:
        - Grade distribution charts
        - Performance cluster plots
        - Benchmark analysis

## Customization
To modify the application:
- Edit `app.py` to change analysis logic
- Adjust `grade_order` variable for different grading systems
- Modify visualization parameters in the plotting functions

## Troubleshooting

### Common Issues
1. **File Format Errors**:
   - Ensure column names match exactly
   - Remove special characters from headers
   - Verify numeric columns contain only numbers or 'ABS'

2. **Visualization Problems**:
   - Large datasets may require plot adjustments
   - Try reducing the number of classes displayed

3. **PDF Generation Failures**:
   - Ensure temporary directory permissions
   - Check for matplotlib backend conflicts

