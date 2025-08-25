# 📊 Student Performance Dashboard

## 📋 Introduction
The Student Performance Dashboard is a comprehensive Streamlit web application designed for educational institutions to analyse and visualise student assessment data. This tool helps educators quickly understand class performance, grade distributions, and gender demographics across multiple classes and year levels..

## ✨ Key Features
- **Multi-Source Data Import**: Upload Excel/CSV files or connect directly to Google Sheets
- **Year-Level Filtering**: Analyse data by specific year levels (e.g., Year 9, Year 10)
- **Comprehensive Grade Analysis**: Visualise grade distributions through bar charts and pie charts
- **Gender Distribution Analysis**: Track male/female student distribution across classes
- **Class Performance Summary**: View statistical summaries of assessment scores
- **Export Capabilities**: Generate PDF reports and download analysed data as CSV
- **Responsive Design**: Clean, professional interface with intuitive navigation


### 📂 Data Format Requirements
Your Excel/CSV file should contain these columns (case-insensitive):
- `Name` - Student names
- `Gender` - Male(M) /Female(F)
- `Class` - Class identifier (e.g., 1A, 2A, 10Sc1)
- `Assessment` - Continuous Assessment percentage (%)
- `Examination` - Exam percentage (%)
- `Total` - Combined score between Assessment and Examination
- `Grade` - Letter grade (A*, A, B, C, D, etc.)
Grade Scale: A*, A, B, C, D, E, U, ABS (absent)

You may name your sheets as the class name in the spreadsheets. The software supports multiple class-sheets in one Excel spreadsheet.

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 (when deployed at Streamlit)
- PIP package manager
- Google Cloud account

### Step 1: Clone/Download the Application and Dependencies
1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd student-performance-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   or run the following manually:
   ```bash
   pip install streamlit pandas matplotlib fpdf gspread google-auth pillow numpy
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

### Step 2: Google Cloud Project Setup (for Google Sheet Integration)
#### 1. Create a Google Cloud Project:
- Go to the Google Cloud Console
- Click "Select a project" → "New Project"
- Name your project (e.g., **"Student-Dashboard"**) and create it

#### 2. Enable Google Sheets API:
- In your project dashboard, go to **"APIs & Services" → "Library"**
- Search for **"Google Sheets API"** and enable it
- Search for **"Google Drive API"** and enable it

#### 3. Create Service Account Credentials:
- Go to **"APIs & Services" → "Credentials"**
- Click **"Create Credentials" → "Service Account"**
- Fill in the service account details (name, description)
- Click **"Create and Continue"**
- Assign the **"Viewer"** role (or more restrictive role as needed)
- Click **"Done"**

#### 4. Generate Credentials JSON:
- Click on your newly created service account
- Go to the **"Keys"** tab
- Click **"Add Key" → "Create New Key"**
- Select "JSON" format and create
- The JSON file will download automatically

#### 5. Configure Streamlit Secrets:
- **Option A**: Using Streamlit Secrets File (Local Development)
   - Create a .streamlit folder in your project directory
   - Create a **secrets.toml** file inside this folder
   - Add your Google Sheets credentials:
   ```bash
   # .streamlit/secrets.toml
   GSHEET_CREDENTIALS_JSON = '''
   {
   "type": "service_account",
   "project_id": "your-project-id",
   "private_key_id": "your-private-key-id",
   "private_key": "-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n",
   "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
   "client_id": "your-client-id",
   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
   "token_uri": "https://oauth2.googleapis.com/token",
   "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
   "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com"
   }
   '''
   ```

- **Option B**: Using Streamlit Secrets File (Local Development)
   - When deploying to Streamlit Cloud, use the secrets management in the app settings
   - Add the same JSON content as a secret variable named **GSHEET_CREDENTIALS_JSON**

#### 6. Share Your Google Sheet:
- Open your Google Sheet
- Click **"Share"** and add your service account email (from the credentials) as a **"Viewer/Editor"**, or
- Click **"Share"** and change the access to **"Anyone with the link"** as **"Viewer/Editor"**


## 🔧 Troubleshooting
#### Google Sheets Connection Issues:
- Verify your service account has access to the spreadsheet
- Check that the Google Sheets API is enabled in your Google Cloud project
- Ensure the credentials JSON is properly formatted in your secrets

#### File Upload Issues:
- Verify your Excel/CSV file follows the required format
- Check that column names match the expected patterns

#### Visualisation Errors:
- Ensure all required data columns are present
- Check for non-numeric values in assessment/exam columns

## 📄 License
This project is open source and available under the MIT License.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

For questions or support, please contact your system administrator or refer to the Streamlit documentation.
