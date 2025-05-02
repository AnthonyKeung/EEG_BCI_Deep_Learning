from matlab_processor import matlab_to_DL_input
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Conv2D, MaxPooling2D, Flatten, Dense, Reshape, Dropout, LSTM
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt



class ModelEvaluation:
    def __init__(self, subjects, n_folds=5):

        self.n_folds = n_folds
        self.subjects = subjects
        self.results = []
        self.accuracies = []
    
    def CSP_evaluation_of_dataset(self):
        for subject in range(1, self.subjects+1):

            X,Y = matlab_to_DL_input([f"BCI_IV_2a_mat/A0{subject}E.mat", f"BCI_IV_2a_mat/A0{subject}T.mat"],
                                     window_size=500,
                                     number_of_channels = 22, 
                                     sampling_freq = 250, 
                                     filter=True)
            
            # X,Y = matlab_to_DL_input([f"BCI_IV_2b_mat/B0{subject}E.mat", f"BCI_IV_2b_mat/B0{subject}T.mat"],
            #                          window_size=750,
            #                          number_of_channels = 3, 
            #                          sampling_freq = 250, 
            #                          filter=True)

            
            #Kfold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            accuracies = []
            precision_scores  = []
            recalls = []
            f1s = []
            
            for train_index, test_index in kf.split(X, Y):
                n_components = 6
                csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
                rf = RandomForestClassifier(n_estimators=500, max_depth=200, random_state=42, criterion="entropy")
                X = X.astype('float64')

                # Convert Y to integer scalar array some more bullshit 
                Y = np.array(Y).astype(int).ravel()

                
                    
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]


                # Apply CSP
                X_train_csp = csp.fit_transform(X_train, Y_train)
                X_test_csp = csp.fit_transform(X_test, Y_test)
                # Train the model
                rf.fit(X_train_csp, Y_train) 

                # Evaluate the model
                Y_pred = rf.predict(X_test_csp)
                accuracy = accuracy_score(Y_test, Y_pred)
                precision = precision_score(Y_test, Y_pred, average='weighted')
                recall = recall_score(Y_test, Y_pred, average='weighted')
                f1 = f1_score(Y_test, Y_pred, average='weighted')

                accuracies.append(accuracy)
                precision_scores.append(precision)
                recalls.append(recall)
                f1s.append(f1)

            self.results.append({     
                "subject": subject,
                "accuracy": np.mean(accuracies),
                "precision": np.mean(precision_scores),
                "recall": np.mean(recall),
                "f1s": np.mean(f1s)
            })
        
        print("Results: ", self.results)

    def CNN_evaluation_of_dataset(self, dataset):
        for subject in range(1, self.subjects+1):

            if dataset == "2a":
                num_channels = 22
                num_timepoints = 500
                num_of_classes = 4
                X, Y = matlab_to_DL_input([f"BCI_IV_2a_mat/A0{subject}E.mat", f"BCI_IV_2a_mat/A0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          filter=False)

            elif dataset == "2b":
                num_timepoints = 750
                num_channels = 3
                num_of_classes = 2
                X, Y = matlab_to_DL_input([f"BCI_IV_2b_mat/B0{subject}E.mat", f"BCI_IV_2b_mat/B0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          filter=False )

            else:
                raise ValueError("Invalid dataset. Choose '2a' or '2b'.")

            # Let's try no normalization for now but still need a 4D tensor         
            Y = np.array(Y).astype(int).ravel() 
            X = X[..., np.newaxis]
            X = np.transpose(X, (0, 2, 1, 3))

            Y = [label - 1 for label in Y] # 769 is the frist label 
            Y = to_categorical(Y, num_classes=num_of_classes)

            # Kfold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            accuracies = []
            precision_scores = []
            recalls = []
            f1s = []

            fold = 0

            for train_index, test_index in kf.split(X, Y):
                print(f"Currently working on Subject {subject}, fold {fold+1}")
                fold += 1 
                # Defining the CNN model
                def create_model(dropout_rate=0.4, alpha=0.1, dense_units=100):
                    model = Sequential([
                        Conv2D(8, (11, 1), activation=None, input_shape=(num_timepoints, num_channels, 1), strides=1, padding='valid'),
                        LeakyReLU(alpha=0.05),
                        Dropout(dropout_rate),

                        Conv2D(16, (11, 1), activation=None, strides=1, padding='valid'),
                        LeakyReLU(alpha=alpha),
                        Dropout(dropout_rate),
                        MaxPooling2D((1, 2)),

                        Flatten(),
                        Dense(dense_units, activation=None),
                        LeakyReLU(alpha=alpha),
                        Dense(num_of_classes, activation='softmax')
                    ])
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                    return model

                # Wrap the model for GridSearchCV
                model = KerasClassifier(build_fn=create_model, verbose=0)

                # Define the hyperparameter grid
                param_grid = {
                    'dropout_rate': [0.3, 0.4, 0.5],
                    'alpha': [0.05, 0.1, 0.2],
                    'dense_units': [50, 100, 150],
                    'batch_size': [16, 32],
                    'epochs': [10, 15]
                }

                # Perform grid search
                grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
                grid_result = grid.fit(X_train, Y_train)

                # Print the best parameters and accuracy
                print(f"Best Parameters: {grid_result.best_params_}")
                print(f"Best Accuracy: {grid_result.best_score_}")

                # Use the best model
                best_model = grid_result.best_estimator_
                test_loss, test_accuracy = best_model.model.evaluate(X_test, Y_test)
                print(f"Test Accuracy: {test_accuracy}")

                # # # Define the CNN model
                # model = Sequential([
                #     Conv2D(25, (11,1), activation=None, input_shape=(num_timepoints, num_channels, 1), strides=1, padding='valid'),
                #     LeakyReLU(alpha=0.05),
                #     Dropout(0.20),
                #     Reshape((3, 740, 25)),

                #     Conv2D(25, (1, 2), activation=None, strides=1, padding='valid'),
                #     LeakyReLU(alpha=0.05),
                #     MaxPooling2D((1, 3)),
                #     Reshape((246, 3, 25)),

                #     Conv2D(25, (11, 1), activation=None, strides=1, padding='valid'),
                #     LeakyReLU(alpha=0.05),
                #     Reshape((3, 236, 25)), 
                #     MaxPooling2D((1, 3), strides=1),
                #     Reshape((234, 3, 25)), 

                #     Conv2D(25, (11, 1), activation=None, strides=1, padding='valid'),
                #     LeakyReLU(alpha=0.05),
                #     Dropout(0.20),
                #     Reshape((3 , 224, 25)), 
                #     MaxPooling2D((1, 3)),
                #     Reshape((74 , 3, 25)), 

                #     Conv2D(25, (11, 1), activation=None, strides=1, padding='valid'),
                #     LeakyReLU(alpha=0.05),
                #     Dropout(0.20),
                #     MaxPooling2D((1, 3)),
                #     Flatten(),
                #     Dense(800, activation='relu'),
                #     Dense(num_of_classes, activation='softmax')
                #     ])
    



                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

    
                # Compile the model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Train the model
                history = model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_test, Y_test))

                # Evaluate the model
                test_loss, test_accuracy = model.evaluate(X_test, Y_test)

                # Plot training and validation loss
                plt.figure(figsize=(8, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Plot training and validation accuracy
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

                accuracies.append(test_accuracy)

            self.results.append({
                "subject": subject,
                "accuracy": np.mean(accuracies),
                # "precision": np.mean(precision_scores) if precision_scores else None,
                # "recall": np.mean(recalls) if recalls else None,
                # "f1s": np.mean(f1s) if f1s else None
            })

        print("Results: ", self.results)
        
    def LSTM_evaluation_of_dataset(self, dataset):
        for subject in range(1, self.subjects + 1):
            if dataset == "2a":
                num_channels = 22
                num_timepoints = 500
                num_of_classes = 4
                X, Y = matlab_to_DL_input([f"BCI_IV_2a_mat/A0{subject}E.mat", f"BCI_IV_2a_mat/A0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          filter=False)

            elif dataset == "2b":
                num_timepoints = 750
                num_channels = 3
                num_of_classes = 2
                X, Y = matlab_to_DL_input([f"BCI_IV_2b_mat/B0{subject}E.mat", f"BCI_IV_2b_mat/B0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          artifact_removal=True, 
                                          filter=True)

            else:
                raise ValueError("Invalid dataset. Choose '2a' or '2b'.")

            # Reshape and normalize the data
            X = np.transpose(X, (0, 2, 1))  # Reshape to (n_samples, time_steps, n_features)
            X = (X - np.mean(X, axis=1, keepdims=True)) / np.std(X, axis=1, keepdims=True)
            Y = [label - 1 for label in Y] 
            Y = to_categorical(Y, num_classes=num_of_classes)

            # KFold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            accuracies = []
            f1_scores = []

            fold = 0
            for train_index, test_index in kf.split(X, Y):
                print(f"Currently working on Subject {subject}, fold {fold + 1}")
                fold += 1

                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                # Define the LSTM model
                model = Sequential([
                    LSTM(16, input_shape=(num_timepoints, num_channels), return_sequences=True),
                    Dropout(0.3),
                    LSTM(32, input_shape=(num_timepoints, num_channels), return_sequences=True),
                    Dropout(0.3),
                    LSTM(64, return_sequences=False),
                    Dropout(0.3),
                    Dense(64, activation='relu'),
                    Dropout(0.3),
                    Dense(num_of_classes, activation='softmax')
                ])

                # Define early stopping
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                # Compile the model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Train the model
                history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))

                # Evaluate the model
                test_loss, test_accuracy = model.evaluate(X_test, Y_test)
                f1 = f1_score(np.argmax(Y_test, axis=1), np.argmax(model.predict(X_test), axis=1), average='weighted')
                accuracies.append(test_accuracy)
                f1_scores.append(f1)

                # Plot training and validation loss
                plt.figure(figsize=(8, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Plot training and validation accuracy
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

                # # Generate confusion matrix
                # Y_pred_classes = np.argmax(model.predict(X_test), axis=1)
                # Y_test_classes = np.argmax(Y_test, axis=1)
                # cm = confusion_matrix(Y_test_classes, Y_pred_classes)

                # # Plot confusion matrix
                # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["LH", "RH"])
                # disp.plot(cmap=plt.cm.Blues)
                # plt.title(f"Confusion Matrix for Subject {subject}, Fold {fold}")
                # plt.show()

            self.results.append({
                "subject": subject,
                "accuracy": np.mean(accuracies),
                "f1 Score": np.mean(f1_scores)
            })

        print("Results: ", self.results)

    def DBN_evaluation_of_dataset(self, dataset):
        for subject in range(1, self.subjects + 1):
            if dataset == "2a":
                num_channels = 22
                num_timepoints = 500
                num_of_classes = 4
                X, Y = matlab_to_DL_input([f"BCI_IV_2a_mat/A0{subject}E.mat", f"BCI_IV_2a_mat/A0{subject}T.mat"],
                                        window_size=num_timepoints,
                                        number_of_channels=num_channels,
                                        sampling_freq=250,
                                        filter=False)

            elif dataset == "2b":
                num_timepoints = 750
                num_channels = 3
                num_of_classes = 2
                X, Y = matlab_to_DL_input([f"BCI_IV_2b_mat/B0{subject}E.mat", f"BCI_IV_2b_mat/B0{subject}T.mat"],
                                        window_size=num_timepoints,
                                        number_of_channels=num_channels,
                                        sampling_freq=250,
                                        filter=True)

            else:
                raise ValueError("Invalid dataset. Choose '2a' or '2b'.")

            # Flatten and normalize the data
            X = X.reshape(X.shape[0], -1)  # Flatten to (n_samples, n_features)
            X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            Y = np.array(Y).astype(int).ravel()

            # KFold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            accuracies = []

            fold = 0
            for train_index, test_index in kf.split(X, Y):
                print(f"Currently working on Subject {subject}, fold {fold + 1}")
                fold += 1

                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

                rbm1 = BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=10, random_state=42)
                rbm2 = BernoulliRBM(n_components=128, learning_rate=0.01, n_iter=10, random_state=42)
                logistic = LogisticRegression(max_iter=1000, random_state=42)

                # Create a pipeline
                dbn = Pipeline(steps=[
                    ('rbm1', rbm1),
                    ('rbm2', rbm2),
                    ('logistic', logistic)
                ])

                # Train the DBN
                dbn.fit(X_train, Y_train)

                # Evaluate the DBN
                Y_pred = dbn.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                print(accuracy)
                accuracies.append(accuracy)

            self.results.append({
                "subject": subject,
                "accuracy": np.mean(accuracies)
            })

        print("Results: ", self.results)

    def CNN_LSTM_evaluation_of_dataset(self, dataset):
        for subject in range(1, self.subjects+1):

            if dataset == "2a":
                num_channels = 22
                num_timepoints = 500
                num_of_classes = 4
                X, Y = matlab_to_DL_input([f"BCI_IV_2a_mat/A0{subject}E.mat", f"BCI_IV_2a_mat/A0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          filter=False)

            elif dataset == "2b":
                num_timepoints = 750
                num_channels = 3
                num_of_classes = 2
                X, Y = matlab_to_DL_input([f"BCI_IV_2b_mat/B0{subject}E.mat", f"BCI_IV_2b_mat/B0{subject}T.mat"],
                                          window_size=num_timepoints,
                                          number_of_channels=num_channels,
                                          sampling_freq=250,
                                          filter=True )

            else:
                raise ValueError("Invalid dataset. Choose '2a' or '2b'.")

            # Let's try no normalization for now but still need a 4D tensor         
            Y = np.array(Y).astype(int).ravel() 
            X = X[..., np.newaxis]
            X = np.transpose(X, (0, 2, 1, 3))

            Y = [label - 1 for label in Y] # 769 is the frist label 
            Y = to_categorical(Y, num_classes=num_of_classes)

            # Kfold cross-validation
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            accuracies = []
            
            precision_scores = []
            recalls = []
            f1s = []

            fold = 0

            for train_index, test_index in kf.split(X, Y):
                print(f"Currently working on Subject {subject}, fold {fold+1}")
                fold += 1 
                # Defining the CNN model
                model = Sequential([
                    Conv2D(8, (11, 1), activation=None, input_shape=(num_timepoints, num_channels, 1), strides=1, padding='valid'),
                    LeakyReLU(alpha=0.05),
                    Dropout(0.30),

                    Conv2D(16, (11, 1), activation=None, strides=1, padding='valid'),
                    LeakyReLU(alpha=0.1),
                    Dropout(0.30),
                    MaxPooling2D((1, 2)),

                    Flatten(),
                    Dense(100, activation=None),
                    LeakyReLU(alpha=0.1),
                    Reshape((100, 1)),  # Reshape to feed into LSTM
                    LSTM(64, return_sequences=False),
                    Dense(num_of_classes, activation='softmax')
                ])

                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]

    
                # Compile the model
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                # Train the model
                history = model.fit(X_train, Y_train, epochs=15, batch_size=32, validation_data=(X_test, Y_test))

                # Evaluate the model
                test_loss, test_accuracy = model.evaluate(X_test, Y_test)

                # Plot training and validation loss
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()

                # Plot training and validation accuracy
                plt.figure(figsize=(12, 6))
                plt.plot(history.history['accuracy'], label='Training Accuracy')
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

                accuracies.append(test_accuracy)

            self.results.append({
                "subject": subject,
                "accuracy": np.mean(accuracies),
                # "precision": np.mean(precision_scores) if precision_scores else None,
                # "recall": np.mean(recalls) if recalls else None,
                # "f1s": np.mean(f1s) if f1s else None
            })

        print("Results: ", self.results)
  
if __name__ == "__main__":
    cspEvaluation = ModelEvaluation(1, n_folds=5) ## one subject for now beacuase I'm testing 
    cspEvaluation.LSTM_evaluation_of_dataset(dataset="2a")