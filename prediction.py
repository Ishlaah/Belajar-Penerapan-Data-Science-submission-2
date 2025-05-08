import pandas as pd
import pickle

# Load model yang sudah dilatih
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Masukkan data yang ingin diprediksi
X_new = pd.DataFrame({
    'Application_mode': [15], 
    'Course': [9254], 
    'Previous_qualification_grade': [160.0], 
    'Mothers_qualification': [1], 
    'Fathers_qualification': [3], 
    'Mothers_occupation': [3], 
    'Fathers_occupation': [3], 
    'Admission_grade': [142.5], 
    'Displaced': [1], 
    'Gender': [1], 
    'Scholarship_holder': [0], 
    'Age_at_enrollment': [19], 
    'Curricular_units_1st_sem_enrolled': [6], 
    'Curricular_units_1st_sem_evaluations': [6], 
    'Curricular_units_1st_sem_approved': [6], 
    'Curricular_units_2nd_sem_enrolled': [6], 
    'Curricular_units_2nd_sem_evaluations': [6], 
    'Curricular_units_2nd_sem_approved': [6], 
    'Unemployment_rate': [13.9], 
    'Inflation_rate': [-0.3], 
    'GDP': [0.79], 
    'Status': [1], 
    'Ratio_approved_1st_sem': [1.0], 
    'Ratio_approved_2nd_sem': [1.0]
})

# Model akan memprediksi
y_pred = model.predict(X_new)

# Mengubah nilai prediksi menjadi label yang diminta
y_pred_label = ['Tidak dropout' if pred == '1' else 'Berpotensi dropout' for pred in y_pred]

print("Hasil Prediksi:", y_pred_label)
