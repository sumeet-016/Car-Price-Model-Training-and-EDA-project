# Car Price Prediction — End-to-End Data Analysis & Machine Learning Project

## Project Overview
This project predicts the selling price of used cars using machine learning.  
It includes the full workflow from data cleaning, exploratory data analysis (EDA), feature engineering, model training, evaluation, and model saving.

The final trained model is exported as `Best_Model.pkl`, and the dataset is provided in both raw and cleaned forms.

---

## Project Structure
```
├── car_price_prediction.csv                 # Raw dataset
├── car_price_prediction_updated.csv         # Cleaned & processed dataset
├── ExploratoryDataAnaylsis.ipynb            # EDA and data understanding
├── ModelTraining.ipynb                      # Model building, tuning, evaluation
├── Best_Model.pkl                           # Final trained prediction model
└── README.md
```

---

## Dataset Summary

### Raw Dataset
- File: `car_price_prediction.csv`
- Contains original uncleaned data  
- Issues identified:
  - Missing values
  - Outliers
  - Mixed categorical labels
  - Skewed distributions

### Updated Dataset
- File: `car_price_prediction_updated.csv`
- Cleaned, encoded, and ready for model training
- Includes:
  - Consistent fuel/transmission labels
  - Converted car age
  - Corrected Kms Driven values
  - Encoded categorical variables

---

## Exploratory Data Analysis (EDA)

All EDA steps are documented inside `ExploratoryDataAnaylsis.ipynb`.

### Data Cleaning
- Handling null values  
- Fixing inconsistent categories  
- Removing extreme outliers  
- Converting manufacturing year into car age  

### Feature Engineering
- Created new feature: `Car_Age`  
- Encoded categorical variables:
  - Fuel Type  
  - Seller Type  
  - Transmission  
- Dropped unnecessary or redundant columns  

### Visualization & Insights
Includes:
- Correlation heatmap  
- Price comparison by fuel type  
- Kms Driven vs Selling Price  
- Distribution plots of all numerical variables  

### Key Insights (sample)
- Diesel vehicles tend to have higher resale value than petrol vehicles.  
- Cars older than 6 years show significant depreciation.  
- Automatic transmission generally increases selling price.  
- Higher mileage (70,000+ km) reduces resale value noticeably.

---

## Model Development

The model-building workflow is contained in `ModelTraining.ipynb`.

### Steps Included
- Train-test split  
- Scaling numerical features  
- Encoding categorical features  
- Comparing algorithms such as:
  - Linear Regression
  - Lasso Regression
  - Ridge Regression
  - Random Forest Regressor
- Hyperparameter tuning  
- Selecting the best model based on performance metrics  

### Best Model
- Saved as `Best_Model.pkl`
- Selected on the basis of:
  - R² Score
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

---

## Example: Loading the Model & Predicting

```python
import pickle
import pandas as pd

# Load model
with open('Best_Model.pkl','rb') as f:
    model = pickle.load(f)

# Example input (must match training features)
X_new = pd.DataFrame({
    "Year": [2017],
    "Present_Price": [6.5],
    "Kms_Driven": [32000],
    "Fuel_Type": [1],      
    "Seller_Type": [0],    
    "Transmission": [1],   
    "Owner": [0]
})

# Predict selling price
prediction = model.predict(X_new)
print("Predicted Selling Price:", prediction[0])
```

---

## Technologies Used
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  
- Jupyter Notebook  
- Pickle  

---

## Key Deliverables
- Cleaned and structured dataset  
- Insight-rich EDA with visualizations  
- Regression model for price prediction  
- Exported `.pkl` model for deployment  
- Complete data science workflow  

---

## Future Improvements
- Build a Streamlit or Flask web app for deployment  
- Create a Power BI dashboard for EDA insights  
- Add model explainability (SHAP or feature importance)  
- Improve hyperparameter tuning and try advanced models  
