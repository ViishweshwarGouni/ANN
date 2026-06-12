import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

# One Hot Encoding
y = to_categorical(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save Scaler
joblib.dump(scaler, "scaler.pkl")

# Build ANN Model
model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(4,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    verbose=1
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Accuracy: {accuracy:.4f}")

# Save Model
model.save("iris_ann_model.h5")

print("Model Saved Successfully!")