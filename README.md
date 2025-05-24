# Customer Churn Prediction

A comprehensive machine learning project focused on predicting customer churn using various algorithms and techniques.

## 📊 Project Overview

This project focuses on predicting whether a customer will churn (leave) a service based on various features related to their usage patterns, account details, and service interactions. The project implements a complete ML pipeline from data preprocessing to model deployment, with special attention to handling imbalanced data and model interpretation.

## 📋 Table of Contents
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Key Features](#key-features)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [How to Use](#how-to-use)
- [Contributing](#contributing)

## 📁 Dataset

The dataset contains customer information, usage metrics, and churn status:

- **Customer information**: Account length, international plan, voice mail plan
- **Usage metrics**: Call minutes, charges, number of calls (day, evening, night, international)
- **Service metrics**: Number of voice mail messages, customer service calls
- **Target variable**: Churn (binary)

### Data Files:
- `churn-bigml-20.csv`: Subset with 20% of the data
- `churn-bigml-80.csv`: Subset with 80% of the data

## 🗂️ Project Structure

```
Customer-Churn-Prediction/
│
├── data/
│   ├── churn-bigml-20.csv
│   └── churn-bigml-80.csv
│
├── notebooks/
│   ├── data-prep-final.ipynb    # Data preprocessing and EDA
│   └── model-final.ipynb         # Model training and evaluation
│
└── README.md
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/HamzaChaieb-git/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn xgboost catboost matplotlib seaborn shap
```

## 🔧 Key Features

### Data Preprocessing:
- Missing value imputation using KNN
- Outlier detection and handling
- Feature normalization and scaling
- Class imbalance handling using NearMiss undersampling

### Feature Engineering and Selection:
- Log transformation for skewed features
- Feature importance analysis using multiple methods (Random Forest, Mutual Information, SHAP)
- Selection of optimal feature subset

### Model Development:
- Implementation of 9 machine learning algorithms
- Hyperparameter tuning using GridSearchCV
- K-fold cross-validation for robust evaluation

### Model Interpretation:
- SHAP value analysis for global and local explanations
- Feature importance visualization
- ROC-AUC curve analysis

## 🤖 Models Implemented

1. **Gradient Boosting Classifier**
2. **Random Forest**
3. **XGBoost**
4. **Logistic Regression**
5. **Support Vector Machine (SVM)**
6. **AdaBoost**
7. **Neural Network (MLP)**
8. **CatBoost**
9. **Stacking Ensemble**

## 📈 Results

The models were evaluated using multiple metrics with a focus on ROC-AUC score due to class imbalance. Here are the top-performing models after optimization:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **XGBoost** | **0.93** | **0.92** |
| Random Forest | 0.92 | 0.91 |
| GBM | 0.91 | 0.92 |
| Stacking | 0.92 | 0.91 |
| CatBoost | 0.91 | 0.91 |

## 🔍 Feature Importance

The most influential features for predicting customer churn were:

1. **Total day minutes** - Call duration during the day
2. **Customer service calls** - Number of calls to customer service
3. **International plan** - Whether the customer has an international calling plan
4. **Total international minutes** - Duration of international calls
5. **Total international calls** - Number of international calls made

These key factors indicate that customer service interactions and usage patterns (particularly international and daytime calling) are critical indicators of potential churn.

## 💻 How to Use

### 1. Data Preparation:
Open and run the `notebooks/data-prep-final.ipynb` notebook to:
- Load and explore the dataset
- Perform data cleaning and preprocessing
- Handle missing values and outliers
- Create feature transformations

### 2. Model Training:
Open and run the `notebooks/model-final.ipynb` notebook to:
- Train multiple machine learning models
- Perform hyperparameter tuning
- Evaluate model performance
- Generate predictions

### 3. Making Predictions:
```python
# Example code for making predictions
import pandas as pd
import pickle

# Load the trained model (assuming you've saved it)
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new customer data
new_data = pd.read_csv('new_customer_data.csv')

# Make predictions
predictions = model.predict(new_data)
churn_probabilities = model.predict_proba(new_data)[:, 1]
```

## 🎯 Business Insights

Based on the analysis, here are key recommendations for reducing customer churn:

1. **Proactive Customer Service**: Monitor customers with high service call frequency
2. **International Plan Review**: Optimize international calling plans for better value
3. **Usage Monitoring**: Track customers with unusual calling patterns
4. **Targeted Retention**: Focus on high-usage daytime callers

## 🔮 Future Improvements

- [ ] Implement real-time prediction API
- [ ] Add more advanced ensemble methods
- [ ] Develop customer segmentation analysis
- [ ] Create interactive dashboard for churn monitoring
- [ ] Implement automated model retraining pipeline

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset source: [BigML](https://bigml.com/)
- Inspired by various customer churn prediction projects in the telecommunications industry

---

**Note**: For detailed implementation and code examples, see the Jupyter notebooks in the `notebooks/` directory.
