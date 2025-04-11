# -*- coding: utf-8 -*-


from google.colab import files
uploaded=files.upload()
print(uploaded)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df=pd.read_excel("/content/adult..xlsx")

df.head()

df.tail()

df = df.replace('?', pd.NA)
df = df.dropna()

label_encoders = {}
X = df.drop('income', axis=1)
for column in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df['income'])

income_counts = df['income'].value_counts()


plt.figure(figsize=(6, 6))
plt.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', startangle=90, colors=["lightblue", "salmon"])
plt.title("Income Distribution")
plt.axis('equal')  # Equal aspect ratio makes the pie a circle
plt.show()

corr = X.corr()

plt.figure(figsize=(10, 8))


sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


svm_model = SVC(kernel='rbf')  
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)


print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("Classification Report:\n", classification_report(y_test, y_pred_svm))





