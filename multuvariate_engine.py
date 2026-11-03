import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Create dataset
data = {
    'Horsepower': [130, 165, 150, 150, 140, 198, 220, 215, 225, 190],
    'Weight': [3504, 3693, 3436, 3433, 3449, 4341, 4354, 4312, 4425, 3850],
    'MPG': [18, 15, 18, 16, 17, 15, 14, 14, 14, 15]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Horsepower', 'Weight']]
y = df['MPG']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
print("Actual MPG:", list(y_test))
print("Predicted MPG:", y_pred)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict new vehicle
new_vehicle = pd.DataFrame({
    'Horsepower': [180],
    'Weight': [4000]
})

print("Predicted Fuel Efficiency:", model.predict(new_vehicle)[0])