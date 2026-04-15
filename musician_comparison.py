import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
GENRES = ['Classical', 'HipHop', 'Metal', 'Pop']
N_SAMPLES = 200
N_ELECTRODES = 14

def generate_eeg(genre, n_samples, musician=True):
    """
    Musicians: tighter distributions (scale=0.07), more distinct patterns
    Non-musicians: wider distributions (scale=0.13), more overlap between genres
    """
    #how focused the brain response is (more consistent for musicians)
    scale = 0.07 if musician else 0.13

    if genre == 'Classical':
        alpha = np.random.normal(loc=0.80 if musician else 0.70, scale=scale, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.25 if musician else 0.35, scale=scale, size=(n_samples, N_ELECTRODES))

    elif genre == 'HipHop':
        alpha = np.random.normal(loc=0.50 if musician else 0.52, scale=scale, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.70 if musician else 0.62, scale=scale, size=(n_samples, N_ELECTRODES))

    elif genre == 'Metal':
        alpha = np.random.normal(loc=0.25 if musician else 0.35, scale=scale, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.85 if musician else 0.75, scale=scale, size=(n_samples, N_ELECTRODES))

    elif genre == 'Pop':
        alpha = np.random.normal(loc=0.55 if musician else 0.53, scale=scale, size=(n_samples, N_ELECTRODES))
        beta  = np.random.normal(loc=0.65 if musician else 0.60, scale=scale, size=(n_samples, N_ELECTRODES))

    return np.hstack([alpha, beta])

#musician dataset
X_musician, y_musician = [], []
for i, genre in enumerate(GENRES):
    X_musician.append(generate_eeg(genre, N_SAMPLES, musician=True))
    y_musician.extend([i] * N_SAMPLES)

X_musician = np.vstack(X_musician)
y_musician = np.array(y_musician)

#non-musician dataset
X_nonmusician, y_nonmusician = [], []
for i, genre in enumerate(GENRES):
    X_nonmusician.append(generate_eeg(genre, N_SAMPLES, musician=False))
    y_nonmusician.extend([i] * N_SAMPLES)

X_nonmusician = np.vstack(X_nonmusician)
y_nonmusician = np.array(y_nonmusician)

def build_and_train(X, y, label):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = keras.Sequential([
        keras.layers.Input(shape=(28,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nTraining {label} model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0  # silent training this time
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"{label} model accuracy: {accuracy * 100:.1f}%")

    return model, scaler, X_test, y_test, accuracy, history

#training both
musician_model,    musician_scaler,    X_test_m,  y_test_m,  acc_m,  history_m  = build_and_train(X_musician,    y_musician,    "Musician")
nonmusician_model, nonmusician_scaler, X_test_nm, y_test_nm, acc_nm, history_nm = build_and_train(X_nonmusician, y_nonmusician, "Non-Musician")

#testing the musician model on the non-musician data (without retraining) to see how different the 2 groups are
print("\nCross-group test: Musician model tested on non-musician data...")
X_nm_scaled = musician_scaler.transform(X_nonmusician)
cross_preds  = np.argmax(musician_model.predict(X_nm_scaled), axis=1)
cross_acc    = np.mean(cross_preds == y_nonmusician)
print(f"Cross-group accuracy: {cross_acc * 100:.1f}%")

#accuracy bar comparison chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

bars = axes[0].bar(
    ['Musician\nModel', 'Non-Musician\nModel', 'Cross-Group\nTest'],
    [acc_m * 100, acc_nm * 100, cross_acc * 100],
    color=['steelblue', 'coral', 'mediumpurple'],
    width=0.5
)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Model Accuracy Comparison')
for bar, val in zip(bars, [acc_m * 100, acc_nm * 100, cross_acc * 100]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', fontweight='bold')

#training curves comparison
axes[1].plot(history_m.history['val_accuracy'],  label='Musician',     color='steelblue')
axes[1].plot(history_nm.history['val_accuracy'], label='Non-Musician', color='coral')
axes[1].set_title('Validation Accuracy During Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

#confusion matrices
y_pred_m  = np.argmax(musician_model.predict(X_test_m),  axis=1)
y_pred_nm = np.argmax(nonmusician_model.predict(X_test_nm), axis=1)

cm_m  = confusion_matrix(y_test_m,  y_pred_m)
cm_nm = confusion_matrix(y_test_nm, y_pred_nm)

ConfusionMatrixDisplay(cm_m,  display_labels=GENRES).plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title('Musician Model Predictions')

plt.tight_layout()
plt.savefig('musician_comparison.png')
plt.show()

#non-musician confusion matrix
fig2, ax2 = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(cm_nm, display_labels=GENRES).plot(ax=ax2, colorbar=False, cmap='Oranges')
ax2.set_title('Non-Musician Model Predictions')
plt.tight_layout()
plt.savefig('nonmusician_confusion.png')
plt.show()

print("\nDone! Check musician_comparison.png and nonmusician_confusion.png") #sanity check

#Run with:
#py -3.9 musician_comparison.py