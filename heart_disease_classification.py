import pandas as pd

# Load the processed Excel dataset
df = pd.read_excel("heart_disease_processed.xlsx")

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Handle missing values
df['restecg'] = df['restecg'].fillna('normal')
df['thal'] = df['thal'].fillna('normal')
df['fbs'] = df['fbs'].fillna(False)

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

print("Missing in 'restecg':", df['restecg'].isnull().sum())
df['restecg'] = df['restecg'].fillna(0)

# Check data types and column names
print("\nData info:")
print(df.info())

# Check which columns contain non-numeric data (like "Male", "Female", etc.)
print("\nColumns with non-numeric values:")
print(df.select_dtypes(include=['object']).columns)

from sklearn.preprocessing import LabelEncoder

# Create the label encoder
le = LabelEncoder()

# List of columns to encode
text_columns = ['sex', 'dataset', 'cp', 'restecg', 'slope', 'thal']

# Apply encoding to each column
for col in text_columns:
    df[col] = le.fit_transform(df[col])


# Drop irrelevant columns
df = df.drop(['id', 'dataset'], axis=1)

# Split into features and target
X = df.drop('num', axis=1)
y = df['num']


# Drop columns that are not useful for prediction
df = df.drop(columns=[col for col in ['id', 'dataset'] if col in df.columns])


# Split data into features (X) and target (y)
X = df.drop('num', axis=1)
y = df['num']

# Show shapes of X and y to confirm
print("\nFeatures shape (X):", X.shape)
print("Target shape (y):", y.shape)

import matplotlib.pyplot as plt
import seaborn as sns

# Generate and show the correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

input("Press Enter to continue...")

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict using the test setd
y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_dt = confusion_matrix(y_test, y_pred_dt)    
plt.figure()
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

input("Press Enter to continue...")

# Save accuracy to a variable
accuracy_dt = accuracy_score(y_test, y_pred_dt)




from sklearn.svm import SVC

# Train the SVM model
svm_model = SVC(kernel='linear')  
svm_model.fit(X_train, y_train)

# Make predictions with the test data
y_pred_svm = svm_model.predict(X_test)

# Evaluate the SVM model
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure()
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens')
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

input("Press Enter to continue...")

# Save accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)




from sklearn.neighbors import KNeighborsClassifier

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  
knn_model.fit(X_train, y_train)

# Predict with test set
y_pred_knn = knn_model.predict(X_test)

# Evaluate KNN
print("\nKNN Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nKNN Classification Report:\n", classification_report(y_test, y_pred_dt))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure()
sns.heatmap(cm_knn, annot=True, fmt='d', cmap='YlOrBr')
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

input("Press Enter to continue...")

# Save accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)



from sklearn.naive_bayes import GaussianNB

# Train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict using the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate the model
print("\nNaive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, y_pred_nb))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

cm_nb = confusion_matrix(y_test, y_pred_nb)
plt.figure()
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Purples')
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


input("Press Enter to continue...")

# Save accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)



import matplotlib.pyplot as plt

# Model names and accuracy values
model_names = ['Decision Tree', 'SVM', 'KNN', 'Naive Bayes']
accuracy_values = [accuracy_dt, accuracy_svm, accuracy_knn, accuracy_nb]

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(model_names, accuracy_values, color=['pink', 'blue', 'yellow', 'purple'])

# Add accuracy values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', fontsize=10)

# Final touches
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Classification Models')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

input("Press Enter to continue...")


# Save accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)


from sklearn.naive_bayes import GaussianNB