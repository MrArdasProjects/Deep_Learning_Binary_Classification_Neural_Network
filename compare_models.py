# ==============================================
# Deep Learning Model Comparison
# Dataset: IBM HR Analytics Employee Attrition
# ==============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ----------------------------------------------------
# 1. Load and Preprocess the Dataset
# ----------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Convert target column to binary values
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# One-Hot Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split into features (X) and labels (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------------------------------
# 2. Define Model Configurations
# ----------------------------------------------------
models_config = {
    "Model A (No Dropout)": {
        "layers": [64, 32, 16],
        "dropouts": [],
        "lr": 0.001,
        "batch": 16
    },
    "Model B (Dropout)": {
        "layers": [64, 32],
        "dropouts": [0.3, 0.2],
        "lr": 0.001,
        "batch": 32
    },
    "Model C (Deep + Slow)": {
        "layers": [128, 64, 32],
        "dropouts": [0.4, 0.3, 0.2],
        "lr": 0.0005,
        "batch": 32
    }
}

histories = {}
test_accuracies = {}

# ----------------------------------------------------
# 3. Train All Models
# ----------------------------------------------------
for name, cfg in models_config.items():
    print(f"\n{name} is being trained...")

    model = keras.Sequential()
    for i, units in enumerate(cfg["layers"]):
        if i == 0:
            model.add(layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
        else:
            model.add(layers.Dense(units, activation='relu'))

        if i < len(cfg["dropouts"]):
            model.add(layers.Dropout(cfg["dropouts"][i]))

    model.add(layers.Dense(1, activation='sigmoid'))

    opt = keras.optimizers.Adam(learning_rate=cfg["lr"])
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=cfg["batch"],
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    test_accuracies[name] = acc
    histories[name] = history

    print(f"{name} Test Accuracy: {acc:.4f}")

# ----------------------------------------------------
# 4. Plot Validation Accuracy Comparison
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
for name, hist in histories.items():
    plt.plot(hist.history['val_accuracy'], label=f"{name}")
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig("comparison_accuracy.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------
# 5. Plot Validation Loss Comparison
# ----------------------------------------------------
plt.figure(figsize=(8, 5))
for name, hist in histories.items():
    plt.plot(hist.history['val_loss'], label=f"{name}")
plt.title('Validation Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.legend()
plt.savefig("comparison_loss.png", dpi=300, bbox_inches="tight")
plt.show()

# ----------------------------------------------------
# 6. Display Results
# ----------------------------------------------------
print("\nModel Comparison Results:")
for name, acc in test_accuracies.items():
    print(f"{name}: Test Accuracy = {acc:.4f}")
