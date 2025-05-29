import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_explained_variance, preprocess_data
from utils import plot_explained_variance, preprocess_data, encode_categorical_features

# Set the title of the app
st.title("PCA and Logistic Regression on Adult Income Dataset")

# Load the dataset with Streamlit's cache
@st.cache_data
def load_data():
    data = pd.read_csv(r'E:\FSDS&AI\adult\adult.csv', encoding='latin-1')
    return data

df = load_data()

# Display the dataset
st.subheader("Dataset Preview")
st.write(df.head())


# ...after preprocessing...
df_clean = preprocess_data(df)
df_clean = encode_categorical_features(df_clean)  
X = df_clean.drop('income', axis=1)
y = df_clean['income']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA implementation
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Plot explained variance
st.subheader("Explained Variance Ratio")
plot_explained_variance(pca)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train_pca, y_train)

# Predictions
y_pred = logreg.predict(X_test_pca)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.4f}")

# Visualize PCA results
st.subheader("PCA Biplot")
X_pca_df = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
X_pca_df['income'] = y_train.values
fig, ax = plt.subplots()
sns.scatterplot(data=X_pca_df, x='PC1', y='PC2', hue='income', palette='viridis', alpha=0.5, ax=ax)
ax.set_title('PCA Biplot (First Two Components)')
st.pyplot(fig)