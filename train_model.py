import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #function to split data into training and testing sets
from sklearn.preprocessing import StandardScaler #function to normalize the data (used for neural networks)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf #library for building neural networks
from tensorflow import keras #used in tensorflow for building the model

X = np.load('X.npy')
y = np.load('y.npy')

GENRES = ['Classical', 'HipHop', 'Metal', 'Pop']

print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features") #making sure data loaded correctly

#80% of data for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y #stratify makes sure the genre split is balanced
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

#normalizing, scale all features to have mean=0 and std=1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test) #no data leakage, only using trainign data

#building neural network using keras
model = keras.Sequential([
    keras.layers.Input(shape=(28,)), #28 features in the input data
    
    keras.layers.Dense(64, activation='relu'), #starting with 64 neurons
    keras.layers.Dropout(0.3), #drop 30% of neurons to ensure model learns more robust patterns/doesn't overfit
    
    keras.layers.Dense(32, activation='relu'), #narrowing neurons forces network to prioritize most important info
    keras.layers.Dropout(0.3),
    
    keras.layers.Dense(4, activation='softmax') #4 neurons for 4 genres, softmax gives probabilities for each genre
])

model.summary()

model.compile(
    optimizer='adam', #gradient descent algo adjusts the weights to be more accurate
    loss='sparse_categorical_crossentropy', #how wrong predictions are, uses integers (sparse)
    metrics=['accuracy'] #tells program to track accuracy
)


print("\nTraining the model...") #sanity check
history = model.fit(
    X_train, y_train, #runs data set 50 times, processes 32 samples at a time, sets aside 20% of training data for validation
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1 #prints to terminal
) #history stores


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0) #runs the model on the unseen tst samples (160), gives accuracy and loss
print(f"\nTest Accuracy: {test_accuracy * 100:.1f}%")

#plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#accuracy over epochs
axes[0].plot(history.history['accuracy'],     label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[0].set_title('Model Accuracy Over Training')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

#loss over epochs
axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Model Loss Over Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

#confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=1) #runs probabilities for each genre, takes highest probability's index as predicted genre

cm = confusion_matrix(y_test, y_pred) #compares true vs predicted
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=GENRES)

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Genre Prediction Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

print("\nDone! Check training_history.png and confusion_matrix.png")

#Run with:
#py -3.9 train_model.py