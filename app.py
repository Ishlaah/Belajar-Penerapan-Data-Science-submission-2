import streamlit as st
import pandas as pd
import pickle

# Load model yang sudah dilatih
@st.cache_data()
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Fungsi untuk melakukan prediksi
def predict(model, input_data):
    X_new = pd.DataFrame(input_data, index=[0])
    y_pred = model.predict(X_new)
    return y_pred[0]

# Fungsi utama dashboard
def main():
    # Judul dashboard
    st.title('Prediksi Dropout Mahasiswa')

    # Deskripsi dan informasi input data
    st.header('Masukkan Data untuk Prediksi')
    st.markdown('Isi kolom-kolom berikut dengan data mahasiswa yang ingin Anda prediksi.')

    # Kolom input untuk masukkan data
    application_mode = st.selectbox('Application Mode', [15, 9254])  # Contoh dropdown, ganti dengan pilihan sesuai data Anda
    course = st.selectbox('Course', [9254, 9853])  # Contoh dropdown, ganti dengan pilihan sesuai data Anda
    previous_qualification_grade = st.number_input('Previous Qualification Grade', value=160.0)
    mothers_qualification = st.selectbox('Mother\'s Qualification', [1, 2, 3])
    fathers_qualification = st.selectbox('Father\'s Qualification', [3, 4, 5])
    mothers_occupation = st.selectbox('Mother\'s Occupation', [1, 2, 3])
    fathers_occupation = st.selectbox('Father\'s Occupation', [1, 2, 3])
    admission_grade = st.number_input('Admission Grade', value=142.5)
    displaced = st.selectbox('Displaced', [0, 1])
    gender = st.selectbox('Gender', [0, 1])
    scholarship_holder = st.selectbox('Scholarship Holder', [0, 1])
    age_at_enrollment = st.number_input('Age at Enrollment', value=19)
    curricular_units_1st_sem_enrolled = st.number_input('Curricular Units 1st Sem Enrolled', value=6)
    curricular_units_1st_sem_evaluations = st.number_input('Curricular Units 1st Sem Evaluations', value=6)
    curricular_units_1st_sem_approved = st.number_input('Curricular Units 1st Sem Approved', value=6)
    curricular_units_2nd_sem_enrolled = st.number_input('Curricular Units 2nd Sem Enrolled', value=6)
    curricular_units_2nd_sem_evaluations = st.number_input('Curricular Units 2nd Sem Evaluations', value=6)
    curricular_units_2nd_sem_approved = st.number_input('Curricular Units 2nd Sem Approved', value=6)
    unemployment_rate = st.number_input('Unemployment Rate', value=13.9)
    inflation_rate = st.number_input('Inflation Rate', value=-0.3)
    gdp = st.number_input('GDP', value=0.79)
    status = st.selectbox('Status', [0, 1])
    ratio_approved_1st_sem = st.number_input('Ratio Approved 1st Sem', value=1.0)
    ratio_approved_2nd_sem = st.number_input('Ratio Approved 2nd Sem', value=1.0)

    # Tombol untuk memulai prediksi
    if st.button('Prediksi'):
        # Load model
        model = load_model('model.pkl')

        # Masukkan data ke dalam dictionary
        input_data = {
            'Application_mode': [application_mode],
            'Course': [course],
            'Previous_qualification_grade': [previous_qualification_grade],
            'Mothers_qualification': [mothers_qualification],
            'Fathers_qualification': [fathers_qualification],
            'Mothers_occupation': [mothers_occupation],
            'Fathers_occupation': [fathers_occupation],
            'Admission_grade': [admission_grade],
            'Displaced': [displaced],
            'Gender': [gender],
            'Scholarship_holder': [scholarship_holder],
            'Age_at_enrollment': [age_at_enrollment],
            'Curricular_units_1st_sem_enrolled': [curricular_units_1st_sem_enrolled],
            'Curricular_units_1st_sem_evaluations': [curricular_units_1st_sem_evaluations],
            'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
            'Curricular_units_2nd_sem_enrolled': [curricular_units_2nd_sem_enrolled],
            'Curricular_units_2nd_sem_evaluations': [curricular_units_2nd_sem_evaluations],
            'Curricular_units_2nd_sem_approved': [curricular_units_2nd_sem_approved],
            'Unemployment_rate': [unemployment_rate],
            'Inflation_rate': [inflation_rate],
            'GDP': [gdp],
            'Status': [status],
            'Ratio_approved_1st_sem': [ratio_approved_1st_sem],
            'Ratio_approved_2nd_sem': [ratio_approved_2nd_sem]
        }

        # Lakukan prediksi
        prediction = predict(model, input_data)

        # Tampilkan hasil prediksi
        if prediction == '1':
            st.success('Hasil Prediksi: Tidak dropout')
        else:
            st.warning('Hasil Prediksi: Berpotensi dropout')

# Memanggil fungsi utama untuk menjalankan dashboard
if __name__ == '__main__':
    main()
