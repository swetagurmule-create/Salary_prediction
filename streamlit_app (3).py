
import streamlit as st
import pickle
import gzip
import pandas as pd

# Load the compressed model
@st.cache_resource
def load_model(path):
    with gzip.open(path, 'rb') as f:
        model = pickle.load(f)
    return model

model_path = 'Random_Forest_Regressor_compressed.pkl.gz'
model = load_model(model_path)

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for features
rating = st.number_input('Rating (e.g., 3.8)', min_value=1.0, max_value=5.0, value=3.8, step=0.1)

# For Company Name and Job Title, we'll ask for encoded numerical input
st.markdown("**Note:** For 'Company Name' and 'Job Title', please enter the numerical value if you know the encoding from the training data.")
company_name_encoded = st.number_input('Company Name (Encoded, e.g., 8129 for Sasken)', min_value=0, value=8129, step=1)
job_title_encoded = st.number_input('Job Title (Encoded, e.g., 28 for Android Developer)', min_value=0, value=28, step=1)

salaries_reported = st.number_input('Salaries Reported (e.g., 3)', min_value=1, value=3, step=1)

# Mappings for Location, Employment Status, Job Roles (based on observed encodings)
location_mapping = {'Bangalore': 0, 'Hyderabad': 1, 'New Delhi': 2, 'Chennai': 3, 'Mumbai': 4, 'Pune': 5, 'Kolkata': 6, 'Ahmedabad': 7, 'Gurgaon': 8, 'Noida': 9, 'Other': 10}
employment_status_mapping = {'Full Time': 1, 'Part Time': 0}
job_roles_mapping = {'Android': 0, 'Backend': 1, 'Data Scientist': 2, 'Deep Learning': 3, 'Frontend': 4, 'Full Stack': 5, 'Java': 6, 'Machine Learning': 7, 'Python': 8, 'Web': 10, 'Other': 9} # Added 'Other' for unknown

location_input = st.selectbox('Location', options=list(location_mapping.keys()), index=0)
employment_status_input = st.selectbox('Employment Status', options=list(employment_status_mapping.keys()), index=0)
job_roles_input = st.selectbox('Job Roles', options=list(job_roles_mapping.keys()), index=0)

# Encode selected categorical values
location_encoded = location_mapping[location_input]
employment_status_encoded = employment_status_mapping[employment_status_input]
job_roles_encoded = job_roles_mapping[job_roles_input]

if st.button('Predict Salary'):
    # Create a DataFrame for prediction
    input_data = pd.DataFrame([{
        'Rating': rating,
        'Company Name': company_name_encoded,
        'Job Title': job_title_encoded,
        'Salaries Reported': salaries_reported,
        'Location': location_encoded,
        'Employment Status': employment_status_encoded,
        'Job Roles': job_roles_encoded
    }])

    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ₹{prediction:,.2f}')
