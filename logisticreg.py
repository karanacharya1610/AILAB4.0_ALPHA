
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('full_dataset.csv', delimiter=',')

# Display columns to check
print(df.columns)

# Define features and target
X = df[['T3', 'TT4', 'TSH', 'goitre']]
y = df['classes']

print(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(y_train.shape)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

# Compute confusion matrix
confu = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy_percentage:.2f}%")
print("Confusion Matrix:\n", confu)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confu, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()