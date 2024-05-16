import streamlit as st
import requests
import json

st.title('Model Selection Interface')

# Define the backend URL
backend_url = 'http://127.0.0.1:5000'

# Option to select model
option = st.selectbox('Select Model', ('K-Means', 'Cosine Similarity'))

if option == 'K-Means':
    st.header('K-Means Prediction')
    
    # Input fields for K-Means
    input_data = st.text_input('Enter input data for K-Means model (comma-separated)')
    
    if st.button('Predict'):
        # Convert the input data to a list of floats
        input_data_list = [float(i) for i in input_data.split(',')]
        response = requests.post(f'{backend_url}/predict_kmeans', json=input_data_list)
        prediction = response.json()['prediction']
        st.write('Prediction:', prediction)

elif option == 'Cosine Similarity':
    st.header('Cosine Similarity Calculation')
    
    # Input fields for Cosine Similarity
    index1 = st.number_input('Enter first index', min_value=0, step=1)
    index2 = st.number_input('Enter second index', min_value=0, step=1)
    
    if st.button('Calculate Similarity'):
        response = requests.post(f'{backend_url}/cosine_similarity', json={'index1': index1, 'index2': index2})
        similarity = response.json()['cosine_similarity']
        st.write('Cosine Similarity:', similarity)
