from flask import Flask, render_template, request, url_for, jsonify 
import joblib
import numpy as np
import shap
from datetime import datetime

app = Flask(__name__)

model = joblib.load("final_model.pkl")

short_feature_names = [
    "Binge Drinking",      # _RFBING6
    "Food Insecurity",     # SDHFOOD1
    "BMI Category",        # _BMI5CAT
    "Home Ownership",      # RENTHOM1
    "Age Group",           # _AGE_G
    "Child Count",         # _CHLDCNT
    "Pay Bills Trouble",   # SDHBILLS
    "Education Level",     # _EDUCAG
    "Emotional Support",   # EMTSUPRT
    "Stress Level",        # SDHSTRE1
    "COPD",                # CHCCOPD3
    "Employment Status",   # EMPLOY1
    "Gender",              # _SEX
    "Diabetes",            # DIABETE4
    "Arthritis",           # _DRDXAR2
    "Kidney Disease",      # CHCKDNY2
    "Race",                # _RACEPRV
    "Job Loss/Reduced",    # SDHEMPLY
    "Asthma",              # _CASTHM1
    "Loneliness",          # SDLONELY
    "Life Satisfaction",   # LSATISFY
    "Metro Status",        # _METSTAT
    "Marital Status",      # MARITAL
    "Income Level",        # _INCOMG1
    "Physical Activity",   # _TOTINDA
    "Smoking Status"       # _SMOKER3
]

def get_shap_data(model, input_array, short_feature_names):
    explainer = shap.Explainer(model, feature_names=short_feature_names)
    shap_values = explainer(input_array)
    values = shap_values.values[0]
    features = shap_values.feature_names
    paired = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
    return {
        "features": [x[0] for x in paired],
        "shap_values": [x[1] for x in paired]
    }

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/discovery')
def discovery():
    return render_template('discovery.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    prediction_result = None
    show_plot = False
    submission_time = None
    username = request.form.get('username', 'Guest')

    if request.method == 'POST':
        try:
            # Ambil data input dari form
            _RFBING6 = int(request.form['_RFBING6'])
            SDHFOOD1 = int(request.form['SDHFOOD1'])
            height_cm = float(request.form['height'])
            weight_kg = float(request.form['weight'])
            height_m = height_cm / 100
            _BMI = round(weight_kg / (height_m ** 2), 2)

            if _BMI < 18.5:
                _BMI5CAT = 1  # Underweight
            elif 18.5 <= _BMI <= 24.9:
                _BMI5CAT = 2  # Normal
            elif 25.0 <= _BMI <= 29.9:
                _BMI5CAT = 3  # Overweight
            else:
                _BMI5CAT = 4  # Obese

            RENTHOM1 = int(request.form['RENTHOM1'])
            _AGE_G = int(request.form['_AGE_G'])
            _CHLDCNT = int(request.form['_CHLDCNT'])
            SDHBILLS = int(request.form['SDHBILLS'])
            _EDUCAG = int(request.form['_EDUCAG'])
            EMTSUPRT = int(request.form['EMTSUPRT'])
            SDHSTRE1 = int(request.form['SDHSTRE1'])
            CHCCOPD3 = int(request.form['CHCCOPD3'])
            EMPLOY1 = int(request.form['EMPLOY1'])
            _SEX = int(request.form['_SEX'])
            DIABETE4 = int(request.form['DIABETE4'])
            _DRDXAR2 = int(request.form['_DRDXAR2'])
            CHCKDNY2 = int(request.form['CHCKDNY2'])
            _RACEPRV = int(request.form['_RACEPRV'])
            SDHEMPLY = int(request.form['SDHEMPLY'])
            _CASTHM1 = int(request.form['_CASTHM1'])
            SDLONELY = int(request.form['SDLONELY'])
            LSATISFY = int(request.form['LSATISFY'])
            _METSTAT = int(request.form['_METSTAT'])
            MARITAL = int(request.form['MARITAL'])
            _INCOMG1 = int(request.form['_INCOMG1'])
            _TOTINDA = int(request.form['_TOTINDA'])
            _SMOKER3 = int(request.form['_SMOKER3'])

            input_features = [
                _RFBING6, SDHFOOD1, _BMI5CAT, RENTHOM1,
                _AGE_G, _CHLDCNT, SDHBILLS, _EDUCAG, EMTSUPRT,
                SDHSTRE1, CHCCOPD3, EMPLOY1, _SEX, DIABETE4,
                _DRDXAR2, CHCKDNY2, _RACEPRV, SDHEMPLY, _CASTHM1,
                SDLONELY, LSATISFY, _METSTAT, MARITAL, _INCOMG1,
                _TOTINDA, _SMOKER3
            ]

            input_array = np.array([input_features])

            # JSON default
            response_data = {
                "success": False,
                "result": None,
                "show_plot": False,
                "submission_time": None,
                "username": username,
                "error": None,
                "raw_prediction_value": 0,
                "shap_chart_data": None
            }

            if model:
                raw_prediction = model.predict(input_array)[0]
                show_plot = True
                shap_chart_data = get_shap_data(model, input_array, short_feature_names)
                response_data["shap_chart_data"] = shap_chart_data

                if isinstance(raw_prediction, (int, np.integer, float, np.floating)):
                    if raw_prediction == 1:
                        prediction_result = "Hasil Prediksi: Risiko Tinggi Depresi. Disarankan untuk mencari bantuan profesional."
                        response_data["raw_prediction_value"] = 1
                    elif raw_prediction == 0:
                        prediction_result = "Hasil Prediksi: Risiko Rendah Depresi. Tetap jaga kesehatan mental Anda!"
                        response_data["raw_prediction_value"] = 0
                    else:
                        prediction_result = f"Prediksi Mentah: {raw_prediction}"
                else:
                    prediction_result = raw_prediction

                response_data["success"] = True
                response_data["result"] = prediction_result
                response_data["show_plot"] = show_plot
                response_data["submission_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                response_data["error"] = "Model prediksi belum dimuat."

        except ValueError as e:
            response_data["error"] = f"Terjadi kesalahan input: {e}. Pastikan semua kolom terisi dengan benar."
        except KeyError as e:
            response_data["error"] = f"Data tidak lengkap: Kolom '{e}' tidak ditemukan."
        except Exception as e:
            response_data["error"] = f"Kesalahan tak terduga: {e}"

        return jsonify(response_data)

    return render_template('prediction.html',
                           result=prediction_result,
                           show_plot=show_plot,
                           submission_time=submission_time,
                           username=username,
                           form_data=request.form if request.method == 'POST' else {})

    
if __name__ == '__main__':
    app.run(debug=True)
