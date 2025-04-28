# 🚲 Bike Rental Prediction

This project predicts daily bike rental counts using a **Random Forest Regressor**.  
It uses real-world data from a bike-sharing system, performs data preprocessing, model training, and evaluates performance using MAE, RMSE, and R² Score.

## 📂 Project Structure

- **Data Preparation**:  
  - Load and clean the dataset.
  - Extract useful features like day of week, month, year, temperature, humidity, etc.

- **Model Building**:  
  - Train a Random Forest Regressor on the training set.

- **Evaluation**:  
  - Evaluate predictions using:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - R² Score (goodness of fit)

- **Visualization**:  
  - Plot actual vs predicted bike rental counts for better understanding.

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- KaggleHub (to download dataset)

## 🚀 How to Run

1. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib scikit-learn kagglehub
