# AI-ML-Internship-tasks
# Task 1: Iris Dataset Exploration

## Objective
Perform exploratory data analysis (EDA) on the Iris dataset to understand feature distributions, relationships between variables, and identify outliers.

## Dataset Used
**Iris Dataset** (included in Seaborn)
- 150 samples of iris flowers (50 per species)
- Features:
  - `sepal_length` (cm)
  - `sepal_width` (cm)
  - `petal_length` (cm)
  - `petal_width` (cm)
- Target:
  - `species` (setosa, versicolor, virginica)

## Methods Applied
1. **Data Exploration**
   - Statistical summaries (`.describe()`)
   - Missing value checks
2. **Visualizations**
   - Pairplot (scatter matrix)
   - Histograms
   - Box plots
3. **Tools**
   - Python (Pandas, Seaborn, Matplotlib)
   - Jupyter Notebook

## Key Findings
1. **Species Separation**:
   - Setosa is easily distinguishable by petal measurements
   - Virginica has the largest flowers on average

2. **Feature Importance**:
   - Petal measurements are better classifiers than sepal measurements
   - Petal length shows the clearest separation between species

3. **Outliers**:
   - Few outliers in sepal_width for setosa
   - Virginica shows the widest range in petal sizes

## Sample Visualization
![pairplot](https://github.com/user-attachments/assets/eee79a2a-915f-4119-9a72-a0dbe9b705db)
)

## How to Run
1. Install requirements:
   ```bash
   pip install pandas seaborn matplotlib jupyter


```markdown
# Task 2 : Health Assistant Chatbot

A Python-based chatbot that provides general health information using Mistral-7B-Instruct LLM with safety filters and GUI interface.

# Features
- LLM Integration with Mistral-7B-Instruct
- Safety mechanisms for emergencies
- Tkinter GUI interface
- Free Hugging Face API usage
- Ethical prompt engineering

## Installation
```bash
pip install requests
```

## Usage
1. Get free API key from [Hugging Face](https://huggingface.co/settings/tokens)
2. Replace `YOUR_HUGGING_FACE_API_KEY` in code
3. Run:
```bash
python health_assistant.py
```

## Technical Stack
- Python 3.x
- Tkinter (GUI)
- Hugging Face Inference API
- Mistral-7B-Instruct model

## Safety Features
- Detects self-harm/suicide risks
- Identifies serious medical conditions
- Avoids medical advice/diagnoses

## Example Queries
 "What causes sore throat?"  
 "Is paracetamol safe for children?"  
 "I'm having chest pain" (triggers emergency response)

## Limitations
- Not for medical diagnosis
- API rate limits apply
- Basic symptom checking only

Developed as part of my internship task
```

This version is:
1. Concise but comprehensive
2. Clearly organized with headers
3. Contains all key technical details
4. Highlights safety features
5. Includes clear usage instructions

```
**House Price Prediction using Machine Learning**

**Task Objective**
Build a regression model to predict house prices based on features like area, bedrooms, location, and condition. Compare multiple models and visualize the results.

 **Dataset**
- **Filename:** `train.csv`  
- **Target Variable:** `Price`  
- **Features Used:**
  - Area  
  - Bedrooms  
  - Bathrooms  
  - Floors  
  - YearBuilt  
  - Location  
  - Condition  
  - Garage

**Models Applied**
- Linear Regression  
- Gradient Boosting Regressor

## ⚙️ Workflow Summary
1. Load dataset using pandas  
2. Preprocess data: handle missing values, scale, and encode  
3. Train models  
4. Evaluate performance (MAE, RMSE)  
5. Visualize predictions and feature importances  
6. Save best-performing model using joblib

## 📈 Results

| Model              | MAE (USD) | RMSE (USD) |
|-------------------|-----------|------------|
| Linear Regression | XXXX.XX   | XXXX.XX    |
| Gradient Boosting | XXXX.XX ✅ | XXXX.XX ✅  |

_Replace XXXX.XX with actual output after running the script._

## Output Files
-  ![prediction_comparison](https://github.com/user-attachments/assets/9bcb9461-29a0-4f30-a4a3-f7b7347e9cfa) 
- `![feature_importance](https://github.com/user-attachments/assets/67be50d3-5b5c-4896-929d-9ec1d37330bf)
g` – Visual of top features  


##  How to Run
1. Ensure `` is in the same directory as `prediction.py`  
2. Run the following command in terminal:

   ```bash
**python prediction.py**

```

 # **Customer Churn Prediction ML Pipeline**  
**A production-ready machine learning system to predict telecom customer churn**  

---

## **📌 Table of Contents**  
1. [Project Overview](#-project-overview)  
2. [Dataset](#-dataset)  
3. [Technical Approach](#-technical-approach)  
4. [Installation](#-installation)  
5. [Usage](#-usage)  
6. [Results](#-results)  
7. [Deployment](#-deployment)  
8. [Future Improvements](#-future-improvements)  
9. [Contributing](#-contributing)  

---

## **🔍 Project Overview**  
This project builds an **automated ML pipeline** to predict customer churn (cancellation of service) for telecom companies. Key features:  
✔ **End-to-end pipeline** (preprocessing + modeling)  
✔ **Hyperparameter tuning** with `GridSearchCV`  
✔ **Production-ready** (exported as a `.joblib` file)  
✔ **Explainable predictions** (probability scores)  

**Business Impact**: Helps reduce customer attrition by identifying at-risk users early.  

---

## **📂 Dataset**  
**Source**: [IBM Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)  
**Features**:  
- **Numerical**: `tenure`, `MonthlyCharges`, `TotalCharges`  
- **Categorical**: `gender`, `Contract`, `InternetService`, etc.  
- **Target**: `Churn` (Yes/No → 1/0)  

**Preprocessing**:  
- Handled missing values in `TotalCharges`.  
- Encoded categorical variables (`OneHotEncoder`).  
- Scaled numerical features (`StandardScaler`).

---

## **⚙️ Technical Approach**  
### **1. Models Compared**  
| Model               | Accuracy (CV) | Key Hyperparameters Tuned          |
|---------------------|--------------|-----------------------------------|
| Logistic Regression | ~79%         | `C`, `penalty`, `solver`          |
| **Random Forest**   | **~80%**     | `n_estimators`, `max_depth`       |

### **2. Pipeline Architecture**  
```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(), categorical_features)
    ])),
    ("classifier", RandomForestClassifier())
])
```

### **3. Optimization**  
- **GridSearchCV**: 5-fold cross-validation.  
- **Best Model**: Random Forest (`n_estimators=200`, `max_depth=20`).  

---

## **🛠️ Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt  # pandas, scikit-learn, joblib
   ```  

---

## **🚀 Usage**  
### **1. Training the Pipeline**  
Run the Jupyter notebook or Python script:  
```bash
python churn_prediction.py
```  

### **2. Making Predictions**  
Load the saved pipeline and predict on new data:  
```python
from joblib import load
pipeline = load("churn_pipeline.joblib")
prediction = pipeline.predict(new_customer_data)
```  

**Example Input**:  
```python
new_customer_data = pd.DataFrame({
    "tenure": [12],
    "MonthlyCharges": [70.50],
    "TotalCharges": [850.00],
    "gender": ["Female"],
    # ... (all other features)
})
```  

**Output**:  
```
Churn prediction: Yes (Probability: 72%)
```  

---

## **📊 Results**  
| Metric          | Logistic Regression | Random Forest |
|----------------|--------------------|--------------|
| **Accuracy**   | 79.2%              | **80.1%**    |
| **Precision**  | 0.67               | 0.69         |
| **Recall**     | 0.52               | 0.55         |  

**Key Insights**:  
- Customers with **higher monthly charges** are more likely to churn.  
- **Longer tenure** reduces churn risk.  

---

## **🔮 Future Improvements**  
- Add **SHAP values** for explainability.  
- Incorporate **real-time data** (e.g., customer service logs).  
- Deploy on **AWS SageMaker** for scalability.  

---

---  
**Built with**: Python, Scikit-learn, Pandas  
**Author**: Maryam Abdullah  

---  
✨ **Predict churn → Retain customers → Boost revenue!** ✨

  

