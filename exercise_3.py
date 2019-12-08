# Keras uses the Sequential model for linear stacking of layers.
# That is, creating a neural network is as easy as (later)
# defining the layers!
from tensorflow.keras.models import Sequential
# Everything we've talked about in class so far is referred to in
# Keras as a "dense" connection between layers, where every input
# unit connects to a unit in the next layer
# We will go over specific activation functions throughout the class.
from tensorflow.keras.layers import Dense
# SGD is the learning algorithm we will use
from tensorflow.keras.optimizers import SGD


def build_one_output_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # Add a input hidden layer with appropriate input dimension
    # 1+ lines
    model.add(Dense(2**10, input_shape=(2,), activation = 'relu'))
   

    # Add a final output layer with 1 unit 
    # 1 line
    model.add(Dense(1, activation='sigmoid'))
    
    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=0.9)  #Stochastic gradient descent
    model.compile(loss="binary_crossentropy", optimizer=sgd)
    return model


def build_classification_model():
    model = Sequential()

    ### YOUR CODE HERE ###
    # First add a fully-connected (Dense) hidden layer with appropriate input dimension
    model.add(Dense(10, input_shape=(2,), activation = 'relu'))

    # Now our second hidden layer 
    model.add(Dense(5, input_shape=(10,), activation = 'relu'))
    

    # Finally, add a readout layer
    model.add(Dense(2, activation = 'softmax'))    

    ######################

    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd, metrics=["accuracy"])
    return model


def build_final_model():
    model = Sequential()
    ### YOUR CODE HERE ###

    model.add(Dense(1024, input_shape=(2,), activation = 'relu'))

    model.add(Dense(5, input_shape=(1024,), activation = 'relu'))
    
    model.add(Dense(2, activation = 'softmax'))  
   
    ######################
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9)  # Stochastic gradient descent
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    # we'll have the categorical crossentropy as the loss function
    # we also want the model to automatically calculate accuracy

    return model


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np


def logistic_regression_model(tune_model = False, X_train = None, y_train = None):
    logreg = LogisticRegression()
    if not tune_model:
        return logreg

    param_grid = {
        'penalty' : ['l1', 'l2'],
        'C' : np.logspace(-4, 4, 20),
        'solver' : ['liblinear']
    }


    grid_search = GridSearchCV(estimator = logreg, scoring='f1', param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    
    grid_search.fit(X_train, y_train)
    
    print("Logistic Regession parameters: ", grid_search.best_params_)
    
    return LogisticRegression(**grid_search.best_params_)


def random_forest_model(tune_model = False, X_train = None, y_train = None):    
    rf = RandomForestClassifier(random_state=26)
    if not tune_model:
        return rf
    param_grid = {
        'max_depth': [i for i in range(1, 10, 3)],
        'max_features': ['sqrt'], # Since we only have 2 features at max for the training data
        'min_samples_leaf': [i for i in range(2, 10, 2)],
        'min_samples_split': [i for i in range(2, 15, 2)],
        'n_estimators': [i for i in range(5, 30, 5)]
    }
    grid_search = GridSearchCV(estimator = rf, scoring='f1', param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
    
    grid_search.fit(X_train, y_train)

    print("Random Forest parameters: ", grid_search.best_params_)
    
    return RandomForestClassifier(random_state=26, **grid_search.best_params_)