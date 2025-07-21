
# app.py

from flask import Flask, render_template, request, make_response
import joblib
import numpy as np
import os
import random
from fpdf import FPDF
from flask import make_response, request

app = Flask(__name__)

# --- Filenames ---
MODEL_FILENAME = 'salary_model.pkl'
SCALER_FILENAME = 'scaler1.pkl'
LABEL_ENCODERS_FILENAME = 'label_encoders.pkl'

# --- Load ML Assets ---
def load_asset(filename):
    if os.path.exists(filename):
        print(f"✅ Loading {filename}...")
        return joblib.load(filename)
    else:
        print(f"🚨 WARNING: File not found - {filename}.")
        return None

model = load_asset(MODEL_FILENAME)
scaler = load_asset(SCALER_FILENAME)
label_encoders = load_asset(LABEL_ENCODERS_FILENAME)

# --- ROUTES ---

@app.route('/')
def home():
    """Renders the main landing page."""
    return render_template('index.html')

@app.route("/about-tech")
def about_tech():
    return render_template("about_tech.html")

@app.route('/predict_form')
def predict_form():
    """Renders the form page for salary prediction."""
    # ---------- THIS IS THE CORRECTED LINE ----------
    # We must pass an empty form_data dictionary on the initial load so the template
    # doesn't crash when trying to access it.
    return render_template('predict_form.html', form_data={})
    # ----------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission, predicts salary, and returns the result."""
    form_data = request.form

    if not all([model, scaler, label_encoders]):
        error_msg = "Prediction server not configured. Please check server logs for missing files."
        return render_template('predict_form.html', error_text=error_msg, form_data=form_data)

    try:
        input_data = {
            'Age': float(form_data.get('Age')),
            'Gender': form_data.get('Gender'),
            'Education Level': form_data.get('Education Level'),
            'Job Title': form_data.get('Job Title'),
            'Years of Experience': float(form_data.get('Years of Experience'))
        }
        gender_encoded = label_encoders['Gender'].transform([input_data['Gender']])[0]
        education_encoded = label_encoders['Education Level'].transform([input_data['Education Level']])[0]
        job_title_encoded = label_encoders['Job Title'].transform([input_data['Job Title']])[0]
        
        input_features = np.array([[
            input_data['Age'], gender_encoded, education_encoded, 
            job_title_encoded, input_data['Years of Experience']
        ]])
        
        scaled_features = scaler.transform(input_features)
        predicted_salary_usd = model.predict(scaled_features)[0]
        predicted_salary_inr = predicted_salary_usd * 83.3
        final_prediction = max(0, predicted_salary_inr + random.uniform(-2500, 2500))
        lakhs_pa = final_prediction / 100000
        result_text = f"₹ {lakhs_pa:.2f} Lakhs p.a."
        
        return render_template('predict_form.html', 
                               prediction_text=result_text, 
                               form_data=form_data)
    except Exception as e:
        return render_template('predict_form.html', error_text=f"An error occurred: {e}", form_data=form_data)

from flask import make_response, request
from fpdf import FPDF

from flask import make_response, request
from fpdf import FPDF

@app.route('/download_report')
def download_report():
    try:
        # Collect all details from the request
        report_data = {
            "prediction": request.args.get('prediction', 'N/A'),
            "age": request.args.get('age', 'N/A'),
            "gender": request.args.get('gender', 'N/A'),
            "education": request.args.get('education', 'N/A'),
            "job_title": request.args.get('job_title', 'N/A'),
            "experience": request.args.get('experience', 'N/A'),
            "country": request.args.get('country', 'N/A'),
            "industry": request.args.get('industry', 'N/A'),
            "company_size": request.args.get('company_size', 'N/A'),
            "employment_type": request.args.get('employment_type', 'N/A'),
            "remote": request.args.get('remote', 'N/A'),
            "skills": request.args.get('skills', 'N/A'),
            "certifications": request.args.get('certifications', 'N/A'),
            "relocate": request.args.get('relocate', 'N/A'),
            "user_name": request.args.get('userName', 'Anonymous'),
            "user_location": request.args.get('userLocation', 'Unknown'),
        }

        class FancyPDF(FPDF):
            def header(self):
                # Main Banner
                self.set_fill_color(76, 201, 240)
                self.rect(0, 0, 210, 28, 'F')
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 24)
                self.set_xy(0, 10)
                self.cell(0, 10, "Salary Prediction Report", align='C', ln=1)
                # SmartPredict AI Stamp
                self.set_font("Helvetica", "I", 11)
                self.set_text_color(255, 255, 255)
                self.set_xy(0, 20)
                self.cell(0, 10, "Data provided by SmartPredict AI", align='C')
                self.set_y(29)

        pdf = FancyPDF()
        pdf.add_page()

        # Prepared for
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(95, 95, 100)
        pdf.set_y(33)
        pdf.cell(0, 9, f"Prepared for: {report_data['user_name']} ({report_data['user_location']})", ln=1, align='C')
        pdf.ln(4)

        # Salary Prediction Card
        y = pdf.get_y()
        pdf.set_x(20)
        pdf.set_draw_color(250, 190, 78)
        pdf.set_fill_color(250, 232, 158)
        pdf.rect(20, y, 170, 32, 'DF')
        pdf.set_xy(20, y + 6)
        pdf.set_text_color(43, 45, 66)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(170, 8, "Your Predicted Annual Salary", ln=1, align='C')
        pdf.set_font("Helvetica", "B", 26)
        pdf.set_text_color(76, 201, 240)
        pdf.cell(170, 12, report_data['prediction'].replace('₹', 'Rs.'), ln=1, align='C')
        pdf.ln(12)

        # Profile Section
        y = pdf.get_y()
        pdf.set_x(14)
        pdf.set_fill_color(236, 245, 252)
        pdf.rect(14, y, 182, 88, 'F')
        pdf.set_y(y + 4)
        pdf.set_x(18)
        pdf.set_text_color(54, 79, 107)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(70, 8, "Candidate Profile", ln=1)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(60, 68, 80)
        profile_rows = [
            ("Name", report_data['user_name']),
            ("Location", report_data['user_location']),
            ("Age", report_data['age']),
            ("Gender", report_data['gender']),
            ("Education", report_data['education']),
            ("Job Title", report_data['job_title']),
            ("Experience", f"{report_data['experience']} years"),
            ("Country", report_data['country']),
            ("Industry", report_data['industry']),
            ("Company Size", report_data['company_size']),
            ("Employment Type", report_data['employment_type']),
            ("Work Setup", report_data['remote']),
            ("Willing to Relocate", report_data['relocate']),
        ]
        for field, value in profile_rows:
            pdf.set_x(18)
            pdf.cell(60, 8, f"{field}:", border=0)
            pdf.cell(0, 8, f"{value}", ln=1, border=0)
        pdf.ln(2)

        # Skills Section Card
        y = pdf.get_y()
        pdf.set_x(14)
        pdf.set_fill_color(225, 250, 255)
        pdf.rect(14, y, 182, 16, 'F')
        pdf.set_y(y + 3)
        pdf.set_x(18)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(24, 117, 163)
        pdf.cell(0, 7, "Key Skills", ln=1)
        pdf.set_x(18)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(33, 40, 49)
        pdf.multi_cell(174, 7, report_data['skills'])
        pdf.ln(2)

        # Certifications Section Card
        y = pdf.get_y()
        pdf.set_x(14)
        pdf.set_fill_color(250, 232, 158)
        pdf.rect(14, y, 182, 16, 'F')
        pdf.set_y(y + 3)
        pdf.set_x(18)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(140, 100, 35)
        pdf.cell(0, 7, "Certifications", ln=1)
        pdf.set_x(18)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(90, 55, 15)
        pdf.multi_cell(174, 7, report_data['certifications'])
        pdf.ln(2)

        # Explanation & Confidentiality Footer
        pdf.set_x(14)
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(104, 123, 164)
        pdf.multi_cell(0, 7, 
    "- Data provided by you via the SmartPredict AI interface.\n"
    "- AI uses machine learning models, salary datasets, and pay equity best practices to forecast likely salaries for similar profiles in your region and industry.\n\n"
    "All information is confidential and generated exclusively for you by SmartPredict AI.",
    align='L'
)


        # Faint "stamp"/credit at the bottom right corner
        pdf.set_xy(-80, 277)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(230, 230, 230)
        pdf.cell(70, 9, "Generated by SmartPredict AI", ln=1, align='R')

        pdf_content = pdf.output(dest='S').encode('latin-1')
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=salary-prediction-report.pdf'
        return response
    except Exception as e:
        return f"<h1>Error Generating PDF</h1><p>An error occurred: {e}</p>"

    try:
        # Collect all details from the request
        report_data = {
            "prediction": request.args.get('prediction', 'N/A'),
            "age": request.args.get('age', 'N/A'),
            "gender": request.args.get('gender', 'N/A'),
            "education": request.args.get('education', 'N/A'),
            "job_title": request.args.get('job_title', 'N/A'),
            "experience": request.args.get('experience', 'N/A'),
            "country": request.args.get('country', 'N/A'),
            "industry": request.args.get('industry', 'N/A'),
            "company_size": request.args.get('company_size', 'N/A'),
            "employment_type": request.args.get('employment_type', 'N/A'),
            "remote": request.args.get('remote', 'N/A'),
            "skills": request.args.get('skills', 'N/A'),
            "certifications": request.args.get('certifications', 'N/A'),
            "relocate": request.args.get('relocate', 'N/A'),
            "user_name": request.args.get('userName', 'Anonymous'),
            "user_location": request.args.get('userLocation', 'Unknown'),
        }

        class FancyPDF(FPDF):
            def header(self):
                # Main Banner
                self.set_fill_color(76, 201, 240)
                self.rect(0, 0, 210, 28, 'F')
                self.set_text_color(255, 255, 255)
                self.set_font("Helvetica", "B", 24)
                self.set_xy(0, 10)
                self.cell(0, 10, "Salary Prediction Report", align='C', ln=1)
                # SmartPredict AI Stamp
                self.set_font("Helvetica", "I", 11)
                self.set_text_color(255, 255, 255)
                self.set_xy(0, 20)
                self.cell(0, 10, "Data provided by SmartPredict AI", align='C')
                self.set_y(29)

        pdf = FancyPDF()
        pdf.add_page()

        # Prepared for
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(95, 95, 100)
        pdf.set_y(33)
        pdf.cell(0, 9, f"Prepared for: {report_data['user_name']} ({report_data['user_location']})", ln=1, align='C')
        pdf.ln(4)

        # Salary Prediction Card
        pdf.set_x(20)
        pdf.set_draw_color(250, 190, 78)
        pdf.set_fill_color(250, 232, 158)
        pdf.rect(20, y, 170, 36, style='DF')
        pdf.set_xy(20, pdf.get_y() + 6)
        pdf.set_text_color(43, 45, 66)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(170, 8, "Your Predicted Annual Salary", ln=1, align='C')
        pdf.set_font("Helvetica", "B", 26)
        pdf.set_text_color(76, 201, 240)
        pdf.cell(170, 12, report_data['prediction'].replace('₹', 'Rs.'), ln=1, align='C')
        pdf.ln(12)

        # Profile Section
        y0 = pdf.get_y()
        pdf.set_x(14)
        pdf.set_fill_color(236, 245, 252)
        ppdf.rect(20, y, 170, 36, style='DF')
        pdf.set_y(y0 + 4)
        pdf.set_x(18)
        pdf.set_text_color(54, 79, 107)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(70, 8, "Candidate Profile", ln=1)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(60, 68, 80)
        profile_rows = [
            ("Name", report_data['user_name']),
            ("Location", report_data['user_location']),
            ("Age", report_data['age']),
            ("Gender", report_data['gender']),
            ("Education", report_data['education']),
            ("Job Title", report_data['job_title']),
            ("Experience", f"{report_data['experience']} years"),
            ("Country", report_data['country']),
            ("Industry", report_data['industry']),
            ("Company Size", report_data['company_size']),
            ("Employment Type", report_data['employment_type']),
            ("Work Setup", report_data['remote']),
            ("Key Skills", report_data['skills']),
            ("Certifications", report_data['certifications']),
            ("Willing to Relocate", report_data['relocate']),
        ]
        for field, value in profile_rows:
            pdf.set_x(18)
            pdf.cell(60, 8, f"{field}:", border=0)
            pdf.cell(0, 8, f"{value}", ln=1, border=0)
        pdf.ln(4)

        # Explanation & Footer
        pdf.set_x(14)
        pdf.set_font("Helvetica", "I", 11)
        pdf.set_text_color(104, 123, 164)
        pdf.multi_cell(0, 7, 
            "• Data provided by you via the SmartPredict AI interface.\n"
            "• AI uses machine learning models, salary datasets, and pay equity best practices to forecast likely salaries for similar profiles in your region and industry.\n\n"
            "All information is confidential and generated exclusively for you by SmartPredict AI.",
            align='L'
        )

        # Faint "stamp" at bottom right as watermark
        pdf.set_xy(-80, 277)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(230, 230, 230)
        pdf.cell(70, 9, "Generated by SmartPredict AI", ln=1, align='R')

        pdf_content = pdf.output(dest='S').encode('latin-1')
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=salary-prediction-report.pdf'
        return response
    except Exception as e:
        return f"<h1>Error Generating PDF</h1><p>An error occurred: {e}</p>"

    try:
        report_data = {
            "prediction": request.args.get('prediction', 'N/A'),
            "age": request.args.get('age', 'N/A'),
            "gender": request.args.get('gender', 'N/A'),
            "education": request.args.get('education', 'N/A'),
            "job_title": request.args.get('job_title', 'N/A'),
            "experience": request.args.get('experience', 'N/A'),
            "country": request.args.get('country', 'N/A'),
            "industry": request.args.get('industry', 'N/A'),
            "company_size": request.args.get('company_size', 'N/A'),
            "employment_type": request.args.get('employment_type', 'N/A'),
            "remote": request.args.get('remote', 'N/A'),
            "skills": request.args.get('skills', 'N/A'),
            "certifications": request.args.get('certifications', 'N/A'),
            "relocate": request.args.get('relocate', 'N/A'),
            "user_name": request.args.get('userName', 'Anonymous'),
            "user_location": request.args.get('userLocation', 'Unknown'),
        }

        # Custom FPDF theme for attractive design
        pdf = FPDF()
        pdf.add_page()

        # Cover banner (can use a colored rectangle as a header background)
        pdf.set_fill_color(76, 201, 240)     # vibrant cyan
        pdf.rect(0, 0, 210, 30, 'F')
        pdf.set_text_color(255,255,255)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_y(10)
        pdf.cell(0, 10, "Salary Prediction Report", ln=True, align='C')
        pdf.set_text_color(68,68,68)
        pdf.set_font("Helvetica", "", 13)
        pdf.set_y(25)
        pdf.cell(0, 8, f"Prepared for: {report_data['user_name']} ({report_data['user_location']})", ln=True, align='C')
        pdf.set_y(40)

        # Section: Salary Highlight
        pdf.set_draw_color(76,201,240)
        pdf.set_fill_color(250, 232, 158)
        pdf.set_text_color(43,45,66)
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 12, "Predicted Annual Salary", ln=True, align='C', fill=1, border=1)
        pdf.set_font("Helvetica", "B", 29)
        pdf.set_text_color(76, 201, 240)
        prediction_text = report_data['prediction'].replace('₹', 'Rs.')
        pdf.cell(0, 20, prediction_text, ln=True, align='C')
        pdf.ln(5)
        pdf.set_text_color(43, 45, 66)

        # Section divider
        pdf.set_draw_color(76,201,240)
        pdf.set_line_width(1)
        y = pdf.get_y()
        pdf.line(15, y, 195, y)
        pdf.ln(4)

        # Section: Employee Profile
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(0, 10, "Employee Profile", ln=True)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_fill_color(242, 244, 255)
        
        def info_row(label, value):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(44, 62, 80)
            pdf.cell(46, 9, f"{label}:", border=0, align='R', fill=1)
            pdf.set_font("Helvetica", "", 12)
            pdf.set_text_color(70, 70, 86)
            pdf.cell(120, 9, f"{value}", border=0, align='L', fill=0)
            pdf.ln(8)

        info_row("Job Title", report_data['job_title'])
        info_row("Country", report_data['country'])
        info_row("Industry", report_data['industry'])
        info_row("Company Size", report_data['company_size'])
        info_row("Employment Type", report_data['employment_type'])
        info_row("Remote/On-site", report_data['remote'])
        info_row("Education", report_data['education'])
        info_row("Experience (years)", report_data['experience'])
        info_row("Age", report_data['age'])
        info_row("Gender", report_data['gender'])
        info_row("Willing to Relocate", report_data['relocate'])

        # Section: Skills and Certifications as shaded boxes
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(204,235,255)
        pdf.set_text_color(54,79,107)
        pdf.cell(0, 9, "Key Skills", ln=True, fill=1)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(33,40,49)
        pdf.multi_cell(0, 8, report_data['skills'], fill=0)
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(250,232,158)
        pdf.set_text_color(54,79,107)
        pdf.cell(0, 9, "Certifications", ln=True, fill=1)
        pdf.set_font("Helvetica", "", 12)
        pdf.set_text_color(33,40,49)
        pdf.multi_cell(0, 8, report_data['certifications'], fill=0)

        pdf_content = pdf.output(dest='S').encode('latin-1')
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=salary-prediction-report.pdf'
        return response
    except Exception as e:
        return f"<h1>Error Generating PDF</h1><p>An error occurred: {e}</p>"
    try:
        report_data = {
            "prediction": request.args.get('prediction', 'N/A'),
            "age": request.args.get('age', 'N/A'),
            "gender": request.args.get('gender', 'N/A'),
            "education": request.args.get('education', 'N/A'),
            "job_title": request.args.get('job_title', 'N/A'),
            "experience": request.args.get('experience', 'N/A'),
            "user_name": request.args.get('userName', 'Anonymous'),
            "user_location": request.args.get('userLocation', 'Unknown'),
        }

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 10, "Salary Prediction Report", ln=True, align='C')
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Prepared for: {report_data['user_name']} ({report_data['user_location']})", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 10, "Predicted Annual Salary:", ln=True, align='C')
        pdf.set_font("Helvetica", "B", 28)
        prediction_text = report_data['prediction'].replace('₹', 'Rs.')
        pdf.cell(0, 12, prediction_text, ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Employee Profile", ln=True)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 12)
        pdf.cell(95, 8, f"Age: {report_data['age']}")
        pdf.cell(95, 8, f"Gender: {report_data['gender']}", ln=True)
        pdf.cell(95, 8, f"Education: {report_data['education']}")
        pdf.cell(95, 8, f"Experience: {report_data['experience']} years", ln=True)
        pdf.multi_cell(0, 8, f"Job Title: {report_data['job_title']}")
        
        pdf_content = pdf.output(dest='S').encode('latin-1')
        
        response = make_response(pdf_content)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=salary-prediction-report.pdf'
        return response
    except Exception as e:
        return f"<h1>Error Generating PDF</h1><p>An error occurred: {e}</p>"

if __name__ == '__main__':
    app.run(debug=True)