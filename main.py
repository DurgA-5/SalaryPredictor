# app.py

from flask import Flask, render_template, request, make_response
import joblib
import numpy as np
import os
import random
from fpdf import FPDF

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

@app.route('/download_report')
def download_report():
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