"""from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load K-Means model from its pickle file
with open('customer_item_matrix.pkl', 'rb') as file:
    kmeans_model = pickle.load(file)

# Load Cosine Similarity matrix from its pickle file
with open('cosine_similarity.pkl', 'rb') as file:
    cosine_similarity = pickle.load(file)

@app.route('/predict_kmeans', methods=['POST'])
def predict_kmeans():
    data = request.json
    # Assuming 'data' is a JSON object with the necessary input for prediction
    prediction = kmeans_model.predict([data])
    return jsonify({'prediction': prediction.tolist()})

@app.route('/cosine_similarity', methods=['POST'])
def get_cosine_similarity():
    data = request.json
    # Assuming 'data' contains the indices or vectors to compute cosine similarity
    index1 = data['index1']
    index2 = data['index2']
    # Calculate cosine similarity
    similarity = cosine_similarity[index1, index2]
    return jsonify({'cosine_similarity': similarity})

if __name__ == '__main__':
    app.run(debug=True)
"""
import streamlit as st
import pandas as pd
import pickle

# Function to recommend items for a given customer ID
def recommend_items(customer_id):
    try:
        # Load the cosine similarity matrix from the pickle file
        with open('cosine_similarity.pkl', 'rb') as f:
            cosine_similarity = pickle.load(f)

        # Load the customer-item matrix from the pickle file
        with open('customer_item_matrix.pkl', 'rb') as f:
            customer_item_matrix = pickle.load(f)
        
        # Load the item descriptions from the original dataset
        df = pd.read_csv('shopping.csv')  # Adjust the file path as needed
        
        # Check if the customer_id is in the cosine similarity matrix
        if customer_id not in cosine_similarity.index:
            raise ValueError("Customer ID not found in cosine similarity matrix")
        
        # Find the most similar customer based on cosine similarity
        similar_customer = cosine_similarity.loc[customer_id].sort_values(ascending=False).index[1]
        
        # Get the items purchased by the input customer
        items_customer_id = set(customer_item_matrix.loc[customer_id].iloc[customer_item_matrix.loc[customer_id].to_numpy().nonzero()].index)
        
        # Get the items purchased by the similar customer
        items_similar_customer = set(customer_item_matrix.loc[similar_customer].iloc[customer_item_matrix.loc[similar_customer].to_numpy().nonzero()].index)
        
        # Find the items recommended to the input customer
        recommended_items = list(items_similar_customer - items_customer_id)
        
        # Create a DataFrame with recommended items
        recommended_items_df = pd.DataFrame({
            'Description': [df.loc[df['StockCode'] == item_code, 'Description'].iloc[0] for item_code in recommended_items],
            'StockCode': recommended_items
        })
        
        return recommended_items_df

    except ValueError as e:
        # Specific exception for invalid customer ID
        st.error('Please enter a valid customer ID')
        return pd.DataFrame()  # Return an empty DataFrame in case of error

# Streamlit UI
st.title('Item Recommendation System')

# Input field for customer ID
customer_id = st.text_input('Enter Customer ID')

# Button to trigger recommendation
if st.button('Recommend'):
    # Check if customer ID is provided
    if customer_id:
        # Try to convert customer ID to integer
        try:
            customer_id = int(customer_id)
            # Call recommend_items function
            recommendations = recommend_items(customer_id)
            # Check if the recommendations DataFrame is empty
            if not recommendations.empty:
                # Display recommendations
                st.subheader('Recommended Items')
                st.write(recommendations)
        except ValueError:
            st.error('Please enter a valid customer ID')
    else:
        st.error('Please enter a customer ID')
