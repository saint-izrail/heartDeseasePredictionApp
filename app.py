from flask import Flask, render_template, request
import numpy as np
import joblib
import os  
app = Flask(__name__)

# Load the models and scaler
scaler = joblib.load('models/scaler.joblib')
logregmodel = joblib.load('models/logregmodel.joblib')
knnmodel = joblib.load('models/knnmodel.joblib')

# Dictionary untuk menyimpan akurasi model
model_accuracy = {
    'logregmodel': 0.9457,
    'knnmodel': 0.9239
}

# Dictionary untuk parameter information
parameters = {
    'Age': {
        'name': 'Age (Usia)',
        'description': 'Usia pasien dalam tahun',
        'measurement': 'Dapat diukur sendiri',
        'reference': 'N/A'
    },
    'Sex': {
        'name': 'Sex (Jenis Kelamin)',
        'description': 'Jenis kelamin pasien [M: Laki-laki, F: Perempuan]',
        'measurement': 'Dapat diisi sendiri',
        'reference': 'N/A'
    },
    'ChestPainType': {
        'name': 'Chest Pain Type (Tipe Nyeri Dada)',
        'description': '''Tipe nyeri dada yang dialami:
        - TA (Typical Angina): Nyeri dada khas angina
        - ATA (Atypical Angina): Nyeri dada tidak khas angina
        - NAP (Non-Anginal Pain): Nyeri dada bukan angina
        - ASY (Asymptomatic): Tidak ada gejala''',
        'measurement': 'Perlu konsultasi dengan dokter untuk penentuan yang akurat',
        'reference': 'American Heart Association. (2021). Angina (Chest Pain). https://www.heart.org/en/health-topics/heart-attack/angina-chest-pain'
    },
    'RestingBP': {
        'name': 'Resting Blood Pressure (Tekanan Darah Istirahat)',
        'description': 'Tekanan darah saat istirahat dalam mmHg',
        'measurement': 'Dapat diukur sendiri menggunakan tensimeter digital atau manual',
        'reference': 'American Heart Association. (2023). Understanding Blood Pressure Readings. https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings'
    },
    'Cholesterol': {
        'name': 'Cholesterol (Kolesterol)',
        'description': 'Kadar kolesterol dalam darah (mm/dl)',
        'measurement': 'Perlu pemeriksaan laboratorium',
        'reference': 'National Heart, Lung, and Blood Institute. (2023). Blood Cholesterol. https://www.nhlbi.nih.gov/health-topics/blood-cholesterol'
    },
    'FastingBS': {
        'name': 'Fasting Blood Sugar (Gula Darah Puasa)',
        'description': 'Kadar gula darah setelah puasa minimal 8 jam [1: jika > 120 mg/dl, 0: jika ≤ 120 mg/dl]',
        'measurement': 'Perlu pemeriksaan laboratorium atau dapat menggunakan glucometer pribadi',
        'reference': 'American Diabetes Association. (2023). Understanding Blood Sugar and Control. https://www.diabetes.org/healthy-living/medication-treatments/blood-glucose-testing-and-control'
    },
    'RestingECG': {
        'name': 'Resting ECG (EKG Istirahat)',
        'description': '''Hasil pemeriksaan elektrokardiogram saat istirahat:
        - Normal: Hasil normal
        - ST: Memiliki kelainan gelombang ST-T
        - LVH: Menunjukkan kemungkinan hipertrofi ventrikel kiri''',
        'measurement': 'Perlu pemeriksaan di fasilitas kesehatan',
        'reference': 'Mayo Clinic. (2023). Electrocardiogram (ECG or EKG). https://www.mayoclinic.org/tests-procedures/ekg/about/pac-20384983'
    },
    'MaxHR': {
        'name': 'Maximum Heart Rate (Denyut Jantung Maksimum)',
        'description': 'Denyut jantung maksimum yang dicapai saat tes beban (60-202 bpm)',
        'measurement': 'Perlu pemeriksaan dengan tes beban di fasilitas kesehatan',
        'reference': 'American Heart Association. (2023). Target Heart Rates Chart. https://www.heart.org/en/healthy-living/fitness/fitness-basics/target-heart-rates'
    },
    'ExerciseAngina': {
        'name': 'Exercise Angina (Angina saat Aktivitas)',
        'description': 'Nyeri dada yang muncul saat beraktivitas [Y: Ya, N: Tidak]',
        'measurement': 'Dapat dinilai sendiri, namun sebaiknya dikonfirmasi oleh dokter',
        'reference': 'American Heart Association. (2023). Angina (Chest Pain). https://www.heart.org/en/health-topics/heart-attack/angina-chest-pain'
    },
    'Oldpeak': {
        'name': 'Oldpeak / ST Depression (Depresi ST)',
        'description': 'Depresi segmen ST yang diukur dalam tes beban',
        'measurement': 'Perlu pemeriksaan EKG beban di fasilitas kesehatan',
        'reference': 'Mayo Clinic. (2023). Stress Test. https://www.mayoclinic.org/tests-procedures/stress-test/about/pac-20385234'
    },
    'ST_Slope': {
        'name': 'ST Slope (Kemiringan ST)',
        'description': '''Kemiringan segmen ST pada puncak latihan:
        - Up: Menanjak
        - Flat: Datar
        - Down: Menurun''',
        'measurement': 'Perlu pemeriksaan EKG beban di fasilitas kesehatan',
        'reference': 'Mayo Clinic. (2023). Stress Test. https://www.mayoclinic.org/tests-procedures/stress-test/about/pac-20385234'
    }
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html', parameters=parameters)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get values from the form
        features = [
            float(request.form['Age']),
            float(request.form['Sex']),
            float(request.form['ChestPainType']),
            float(request.form['RestingBP']),
            float(request.form['Cholesterol']),
            float(request.form['FastingBS']),
            float(request.form['RestingECG']),
            float(request.form['MaxHR']),
            float(request.form['ExerciseAngina']),
            float(request.form['Oldpeak']),
            float(request.form['ST_Slope'])
        ]
        
        # Convert to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)
        
        # Scale the features
        final_features = scaler.transform(final_features)
        
        # Get selected model
        selected_model = request.form['model']
        
        # Make prediction based on selected model
        if selected_model == 'knnmodel':
            prediction = knnmodel.predict(final_features)
            accuracy = model_accuracy['knnmodel']
            model_name = "K-Nearest Neighbors"
        else:
            prediction = logregmodel.predict(final_features)
            accuracy = model_accuracy['logregmodel']
            model_name = "Logistic Regression"

        output = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        return render_template('result.html', 
                             prediction=output, 
                             accuracy=accuracy,
                             model_name=model_name)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)  
