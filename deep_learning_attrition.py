# ==============================================
# Deep Learning Binary Classification Project
# Dataset: IBM HR Analytics Employee Attrition
# ==============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ----------------------------------------------
# 1. Load and Inspect Dataset
# ----------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing value check:")
print(df.isnull().sum())

# ----------------------------------------------
# 2. Data Preprocessing
# ----------------------------------------------

# Convert target column 'Attrition' into binary values
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Identify categorical columns and apply One-Hot Encoding
categorical_cols = df.select_dtypes(include=['object']).columns
print("\nCategorical columns:", list(categorical_cols))

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nShape after One-Hot Encoding:", df_encoded.shape)

# Split into features (X) and target (y)
X = df_encoded.drop('Attrition', axis=1)
y = df_encoded['Attrition']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData normalization completed.")

# ----------------------------------------------
# 3. Model Definition
# ----------------------------------------------

# Define neural network with dropout regularization
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Adam optimizer with moderate learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Early stopping configuration
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

print("\nModel Summary:")
model.summary()

# ----------------------------------------------
# 4. Model Training
# ----------------------------------------------
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# ----------------------------------------------
# 5. Model Evaluation
# ----------------------------------------------
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {accuracy:.4f}")

# ----------------------------------------------
# 6. Visualize Learning Curves
# ----------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("final_accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("final_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------
# 7. Predictions and Metrics
# ----------------------------------------------

# Print first 10 predictions as integers
probs = model.predict(X_test[:10])
preds = (probs > 0.5).astype(int).reshape(-1).tolist()
truth = [int(v) for v in y_test.values[:10]]
print("\nFirst 10 samples:")
print("True Labels:", truth)
print("Predicted Labels:", preds)

# Evaluate model using additional metrics
y_proba = model.predict(X_test).ravel()
y_pred = (y_proba > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3))

roc_auc = roc_auc_score(y_test, y_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}")
