# **🤖 SVM Classification App with Streamlit**

An interactive web application for training and evaluating Support Vector Machine (SVM) models on the Iris dataset.

## **📋 Features**

* **Interactive Model Training**: Adjust SVM hyperparameters (kernel type, C value, test size)  
* **Real-time Evaluation**: View accuracy metrics and confusion matrix  
* **Live Predictions**: Input flower measurements and get instant species predictions  
* **Data Visualization**: Interactive plots using Plotly  
* **Model Persistence**: Trained models are saved for future predictions

## **🚀 Quick Start**

### **Local Installation**

1. **Clone the repository**

git clone https://github.com/yourusername/svm-streamlit-app.git  
cd svm-streamlit-app

2. **Install dependencies**

pip install \-r requirements.txt

3. **Run the app**

streamlit run app.py

4. **Open in browser** The app will automatically open at `http://localhost:8501`

## **📦 Dataset**

The app uses the **Iris Dataset** which contains:

* 150 samples  
* 4 features (sepal length, sepal width, petal length, petal width)  
* 3 classes (Setosa, Versicolor, Virginica)

## **🎛️ How to Use**

1. **Adjust Parameters** in the sidebar:

   * Select kernel type (linear, rbf, poly, sigmoid)  
   * Adjust C (regularization parameter)  
   * Set test/train split ratio  
2. **Train Model**: Click "Train Model" button

3. **View Results**:

   * Check accuracy metrics  
   * Examine confusion matrix  
   * Read detailed classification report  
4. **Make Predictions**:

   * Adjust the sliders for flower measurements  
   * Click "Predict Species"  
   * View the predicted species and visualization

## **🌐 Deployment Options**

### **Streamlit Cloud (Recommended)**

1. Push your code to GitHub  
2. Go to [share.streamlit.io](https://share.streamlit.io/)  
3. Connect your GitHub repository  
4. Deploy with one click\!

### **Heroku**

1. Create a `Procfile`:

web: sh setup.sh && streamlit run app.py

2. Create `setup.sh`:

mkdir \-p \~/.streamlit/  
echo "\\  
\[server\]\\n\\  
headless \= true\\n\\  
port \= $PORT\\n\\  
enableCORS \= false\\n\\  
\\n\\  
" \> \~/.streamlit/config.toml

3. Deploy:

heroku create your-app-name  
git push heroku main

## **📊 Model Performance**

The SVM model typically achieves:

* **Accuracy**: 95-100% on test set  
* **Best Kernel**: RBF (Radial Basis Function)  
* **Optimal C value**: Around 1.0

## **🛠️ Technologies Used**

* **Streamlit**: Web framework  
* **Scikit-learn**: Machine learning library  
* **Plotly**: Interactive visualizations  
* **Pandas & NumPy**: Data manipulation

## **📝 Project Structure**

svm-streamlit-app/  
├── app.py              \# Main application  
├── requirements.txt    \# Python dependencies  
├── README.md          \# Documentation  
├── svm\_model.pkl      \# Saved model (generated)  
└── scaler.pkl         \# Saved scaler (generated)

## **🤝 Contributing**

Feel free to fork this repository and submit pull requests\!

## **📄 License**

MIT License \- feel free to use this project for learning and development.

## **👤 Author**

Your Name \- [GitHub](https://github.com/yourusername)

---

Made with ❤️ using Streamlit and Scikit-learn

