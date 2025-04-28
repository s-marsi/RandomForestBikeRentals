# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import kagglehub

# Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("lakshmi25npathi/bike-sharing-dataset")
print("Dataset downloaded at:", dataset_path)


# Load the dataset
df = pd.read_csv(f"{dataset_path}/day.csv")

print(f"\n-------------------------------- Data ------------------------------\n\n")
print(df)
print(f"\n\n--------------------------------------------------------------\n\n")

# Prepare the data
df['datetime'] = pd.to_datetime(df['dteday'])  # Convert to datetime
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Choose features (inputs) and target (output)
features = ['day_of_week', 'month', 'year', 'temp', 'hum', 'windspeed', 'season']
target = 'cnt'

X = df[features]  # Input data
y = df[target]    # Target data (bike rentals)

# Split data into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create and train the Random Forest model with OOB validation
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    max_features=0.8,
    oob_score=True  # <<--- Added this line to enable Out Of Bag validation
)
model.fit(X_train, y_train)



# Out-of-Bag Score
print(f"\nOut-of-Bag (OOB) R2 Score: {model.oob_score_:.2f}")



# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score on Test Set: {r2:.2f}")

evaluate(y_test, y_pred)

# Plot the results


plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual', linestyle='dashed')
plt.plot(y_pred, label='Predicted', color='blue')
plt.legend()
plt.title("Random Forest Predictions vs Actual Rentals")
plt.xlabel("Observations")
plt.ylabel("Number of Rentals")
plt.show()
