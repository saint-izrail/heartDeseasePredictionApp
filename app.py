from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')

# Dictionary informasi untuk setiap parameter
PARAMETER_INFO = {
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
        'name': 'ST Depression (Depresi ST)',
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
    return render_template('info.html', parameters=PARAMETER_INFO)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form in the correct order
        features = [
            'Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol',
            'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'
        ]
        
        # Collect all input values
        feature_values = []
        for feature in features:
            value = float(request.form.get(feature))
            feature_values.append(value)

        # Convert to numpy array and reshape
        features_array = np.array(feature_values).reshape(1, -1)
        
        # Scale features using the loaded scaler
        features_scaled = scaler.transform(features_array)

        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Prepare output
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease Detected'
        
        return render_template('result.html', prediction_text=result)
                             
    except Exception as e:
        return render_template('result.html', 
                             prediction_text="Error: Please check your inputs and try again")

if __name__ == "__main__":
    app.run(debug=True)