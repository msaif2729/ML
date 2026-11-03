import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset
data = {
    'Age': [18, 20, 22, 25, 19, 21, 24, 23, 26, 20],
    'Time_Spent': [5, 10, 15, 20, 8, 12, 18, 16, 22, 9],
    'Clicked': [0, 0, 1, 1, 0, 1, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Input and output
X = df[['Age', 'Time_Spent']]
y = df['Clicked']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train SVM model
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Actual:", list(y_test))
print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Predict for a new student
new_student = pd.DataFrame({
    'Age': [22],
    'Time_Spent': [15]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Student will click the advertisement")
else:
    print("Student will not click the advertisement")