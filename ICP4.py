import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

path_to_cv = "C:\\Users\\anori\\OneDrive\\Desktop\\Programming\\CS4710 Intro to Machine Learning\\ICP4\\Dataset (5)\\Dataset\\glass.csv"
df = pd.read_csv(path_to_cv)

X = df.drop('Type', axis=1)
y = df['Type']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Predict on the testing set
y_pred_nb = nb_classifier.predict(X_test)
accuracy = nb_classifier.score(X_test, y_test)
classification_rep = classification_report(y_test, y_pred_nb)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_rep)


# Initialize Support Vector Classifier (SVC)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the testing set using SVC
y_pred_svm = svm_model.predict(X_test)
accuracy = svm_model.score(X_test, y_test)
print("Accuracy:", accuracy)
y_pred_acc = svm_model.predict(X_test)
report = classification_report(y_test, y_pred_acc)
print("Classification Report:", report)
# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_nb, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted (Naive Bayes)')
plt.title('Actual vs Predicted (Naive Bayes)')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_svm, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted (SVC)')
plt.title('Actual vs Predicted (SVC)')

plt.tight_layout()
plt.show()
