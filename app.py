import streamlit as st
import pandas as pd
import numpy as np

# Function to normalize the data
def normalize_data(df):
    norm_df = df.copy()
    for col in df.columns[1:]:  # Skip the first column (alternatives)
        norm_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    return norm_df

# Function to calculate the weighted normalized matrix
def weighted_normalized(df, weights):
    return df.iloc[:, 1:].multiply(weights, axis=1)

# Function to calculate the positive and negative ideal solutions (PIS and NIS)
def pis_nis(df, impact):
    pis = df.max(axis=0) if impact == "+" else df.min(axis=0)
    nis = df.min(axis=0) if impact == "+" else df.max(axis=0)
    return pis, nis

# Function to calculate the Euclidean distance from PIS and NIS
def euclidean_distance(df, pis, nis):
    pis_dist = np.sqrt(((df - pis) ** 2).sum(axis=1))
    nis_dist = np.sqrt(((df - nis) ** 2).sum(axis=1))
    return pis_dist, nis_dist

# Function to calculate the WASPAS score
def waspas_score(pis_dist, nis_dist, lambda_val=0.5):
    score = (lambda_val * nis_dist) + ((1 - lambda_val) * pis_dist)
    return score

# Title and input for the Streamlit app
st.title('SustainRank: Big Data-Driven Educational Sustainability with WASPAS')

st.sidebar.header("Upload Data File")

# File uploader for the combined weights and decision matrix in Excel format
uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    # Load the Excel file
    excel_df = pd.read_excel(uploaded_file, sheet_name=None)
    
    # Assuming the first sheet contains the Decision Matrix and second sheet contains Weights
    decision_matrix_df = excel_df.get('Decision Matrix')
    weights_df = excel_df.get('Weights')

    if decision_matrix_df is not None and weights_df is not None:
        st.write("Decision Matrix:")
        st.write(decision_matrix_df)

        st.write("Weights Data:")
        st.write(weights_df)

        # Normalize the data
        norm_df = normalize_data(decision_matrix_df)

        # Get the user-input weights (assume weights are in the second column of the weights data)
        weights = weights_df.iloc[:, 1].values

        # Calculate the weighted normalized matrix
        weighted_norm_df = weighted_normalized(norm_df, weights)
        st.write("Weighted Normalized Matrix", weighted_norm_df)

        # Define impacts for each criterion (e.g., "+" for benefit, "-" for cost)
        impacts = ["+" for _ in range(weighted_norm_df.shape[1])]  # Default impacts as benefit
        pis, nis = pis_nis(weighted_norm_df, impacts[0])

        # Calculate distances from PIS and NIS
        pis_dist, nis_dist = euclidean_distance(weighted_norm_df, pis, nis)
        
        # Calculate the WASPAS score
        waspas_scores = waspas_score(pis_dist, nis_dist)
        
        # Add the scores to the dataframe
        decision_matrix_df["WASPAS Score"] = waspas_scores
        st.write("Final WASPAS Scores", decision_matrix_df)

        # Plot the ranking
        st.bar_chart(decision_matrix_df.set_index('Alternative')['WASPAS Score'].sort_values(ascending=False))

