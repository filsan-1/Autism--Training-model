# Train Classifier [Beginner Style]

## Overview
This script trains a classifier for detecting autism using a dataset. 
We'll simplify the code to make it easier to understand for beginners.

## Importing Libraries
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```

## Loading Data
```python
dataset = pd.read_csv('path/to/your/dataset.csv')
print("Data loaded:", dataset.head())
```

## Preparing the Data
```python
X = dataset.drop('target_column', axis=1)  # Features
Y = dataset['target_column']  # Target variable
```

## Splitting the Data
```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("Data split into training and test sets.")
```

## Initializing the Classifier
```python
classifier = RandomForestClassifier(n_estimators=100)
```

## Training the Classifier
```python
classifier.fit(X_train, Y_train)
print("Classifier trained.")
```

## Making Predictions
```python
predictions = classifier.predict(X_test)
print("Predictions made on test set.")
```

## Evaluating the Model
```python
accuracy = accuracy_score(Y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

## Conclusion
You have successfully trained a classifier using beginner-friendly code!