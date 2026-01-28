# Used Car Resale Price Prediction

## Project Overview
This project builds an **end-to-end machine learning system** to predict the resale price of used cars based on vehicle features.  
The goal is to provide a **data-driven and transparent pricing tool** that reduces subjectivity in the second-hand car market and helps both buyers and sellers make informed decisions.

The project follows the full data science lifecycle:
- Data cleaning and preprocessing  
- Exploratory Data Analysis (EDA)  
- Feature selection  
- Model training and evaluation  
- Model deployment as a web application  

---

## Business Problem
Used car pricing is often inconsistent and subjective, leading to:
- Overpriced vehicles  
- Reduced buyer trust  
- Difficulty comparing similar cars  

This project aims to **predict a fair resale price** using historical vehicle data and machine learning, based on measurable attributes such as engine size, transmission type, fuel type, vehicle age, and kilometres driven.

---

## Dataset
- **Size:** 15 000+ used vehicle records  
- **Features:** Vehicle age, engine size, transmission type, fuel type, kilometres driven, seating capacity, seller type, and more  
- **Target variable:** Selling price  

*Note:* The dataset source and regional context are unknown, so economic and geographic factors were not included.

---

## Data Preprocessing & EDA
The following steps were performed:
- Removed irrelevant and duplicate columns  
- Handled missing values and zero entries  
- Removed outliers using the **Interquartile Range (IQR)** method  
- Transformed skewed numerical features (square root transformation)  
- Encoded categorical variables  
- Scaled numerical features using standardisation  
- Visualised distributions and correlations using Matplotlib and Seaborn  

### Key Insights
- **Engine size** and **transmission type** had the strongest influence on resale price  
- Automatic transmission vehicles were generally priced higher than manual vehicles  

## Feature Selection
- Applied **ANOVA F-test** to identify the most influential predictors  
- Selected features with the highest mutual information scores to improve model performance and reduce noise  

## Model Development
Three regression models were trained and compared:
- Random Forest Regressor  
- Support Vector Regressor (SVR)  
- K-Nearest Neighbours (KNN)  

### Evaluation Metrics
- R² Score  
- Mean Absolute Error (MAE)  
- Root Mean Squared Error (RMSE)  

**Best-performing model:** Random Forest Regressor  
This model provided the best balance between bias and variance and was selected for deployment.



## Deployment
The final model was deployed as an **interactive web application** using **Streamlit**.

### Application Features
- Users can input vehicle characteristics  
- The model predicts an estimated resale price  
- Designed for non-technical users  

🔗 **Live App:**  
https://car-model-prediction-gxrjvgbzfahdsgfenmpfrp.streamlit.app/

---

## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Streamlit  
- Jupyter Notebook  
- GitHub  


## Project Structure
├── data/
│ └── used_car_data.csv
├── notebooks/
│ └── eda_and_model_training.ipynb
├── app/
│ └── app.py
├── model/
│ └── random_forest_model.pkl
├── requirements.txt
└── README.md


## Limitations & Future Improvements
### Limitations
- Dataset lacks regional and economic context  
- High-priced vehicles are underrepresented, leading to prediction bias  
- Vehicle make and model were excluded  

### Future Enhancements
- Incorporate vehicle brand and model  
- Experiment with advanced models (XGBoost, LightGBM)  
- Balance the dataset for high-end vehicles  
- Add confidence intervals to predictions  
- Build a Power BI dashboard for deeper insights  

---

## Author
**Rebecca Cullum**  
Entry-level Data Scientist  
Cape Town, South Africa  
LinkedIn: *http://www.linkedin.com/in/rebecca-cullum-63537b377/*  

