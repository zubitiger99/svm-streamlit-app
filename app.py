import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os

# Page configuration
st.set_page_config(page_title="SVM Classifier", page_icon="🤖", layout="wide")

# Title and description
st.title("🤖 Support Vector Machine Classifier")
st.markdown("### Iris Dataset Classification")
st.markdown("This app trains an SVM model on the Iris dataset and allows you to make predictions.")

# Sidebar for model parameters
st.sidebar.header("Model Parameters")
kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
c_value = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

# Load and prepare data
@st.cache_data
def load_data():
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species_name'] = df['species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
    return df, iris

df, iris = load_data()

# Display dataset
with st.expander("📊 View Dataset"):
    st.dataframe(df.head(10))
    st.write(f"Dataset shape: {df.shape}")
    st.write(f"Classes: {', '.join(iris.target_names)}")

# Train model
@st.cache_resource
def train_model(kernel, c_value, test_size, random_state=42):
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    svm_model = SVC(kernel=kernel, C=c_value, random_state=random_state)
    svm_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = svm_model.predict(X_test_scaled)
    
    return svm_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, y_pred

# Train button
if st.sidebar.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        model, scaler, X_train, X_test, y_train, y_test, y_pred = train_model(
            kernel, c_value, test_size
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Training Samples", len(y_train))
        with col3:
            st.metric("Test Samples", len(y_test))
        
        # Confusion Matrix
        st.subheader("📈 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, 
                       labels=dict(x="Predicted", y="Actual"),
                       x=iris.target_names,
                       y=iris.target_names,
                       color_continuous_scale="Blues",
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(y_test, y_pred, target_names=iris.target_names)
            st.text(report)
        
        # Save model
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        st.success("✅ Model trained and saved successfully!")

# Prediction section
st.markdown("---")
st.subheader("🔮 Make Predictions")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

if st.button("🎯 Predict Species"):
    if os.path.exists('svm_model.pkl') and os.path.exists('scaler.pkl'):
        # Load model
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # Make prediction
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.decision_function(input_scaled)[0] if hasattr(model, 'decision_function') else None
        
        # Display prediction
        st.success(f"### Predicted Species: **{iris.target_names[prediction]}** 🌸")
        
        # Visualize input against dataset
        fig = px.scatter(df, x='sepal length (cm)', y='sepal width (cm)', 
                        color='species_name', 
                        title="Your Input vs Dataset",
                        color_discrete_sequence=['#636EFA', '#EF553B', '#00CC96'])
        fig.add_scatter(x=[sepal_length], y=[sepal_width], 
                       mode='markers', 
                       marker=dict(size=15, symbol='star', color='yellow', line=dict(width=2, color='black')),
                       name='Your Input')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Please train the model first!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Scikit-learn")