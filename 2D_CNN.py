from MI_EEG_Processor import MI_EEG_Processor
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Reshape 
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

if __name__ == '__main__':
    file_paths = ["BCI_IV_2b/B0101T.gdf", "BCI_IV_2b/B0102T.gdf", "BCI_IV_2b/B0103T.gdf"]
    MI_EEG_Data = MI_EEG_Processor(file_paths)
    input_formulated_data, labels = MI_EEG_Data.gdf_to_raw_data_input()

    # Convert labels to categorical format
    labels_categorical = to_categorical(labels, num_classes=2)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(input_formulated_data, labels_categorical, test_size=0.3, random_state=42)

    # Define the CNN model
    model = Sequential([
        Conv2D(25, (11,1), activation='relu', input_shape=(750, 2, 1), strides=1, padding='valid'),
        Reshape((2, 740, 25), input_shape=(740, 2, 25)),
        Conv2D(25, (1, 2), activation='relu', input_shape=(2, 740, 25), strides=1, padding='valid'),
        MaxPooling2D((1, 3), input_shape=(1, 740,  25)),
        Reshape((246, 2, 25), input_shape=(2, 246, 25)),
        Conv2D(25, (11, 1), input_shape = (246, 2, 25), activation='relu', strides=1, padding='valid'),
        Reshape((2, 236, 25), input_shape=(236, 2, 25)), 
        MaxPooling2D((1, 3), strides=1),
        Reshape((234, 2, 25), input_shape=(2, 234, 25)), 
        Conv2D(25, (11, 1), activation='relu', strides=1, padding='valid'),
        Reshape((2 , 224, 25), input_shape=(224, 2, 25)), 
        MaxPooling2D((1, 3)),
        Reshape((74 , 2, 25), input_shape=(2, 74, 25)), 
        Conv2D(25, (11, 1), activation='relu', strides=1, padding='valid'),
        MaxPooling2D((1, 2)),
        Flatten(),
        Dense(800, activation='relu'),
        Dense(2, activation='softmax')
        ])

    # Print the model summary to see the size after each layer
    model.summary()
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))


    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Compute the confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("Confusion Matrix:")
    print(cm)