import pandas as pd
from sklearn.model_selection import train_test_split

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import to_categorical




def data():
    data = pd.read_csv("/home/hasan/Data Set/breast cancer wisconcin/data.csv").drop(['Unnamed: 32', 'id'], axis=1)
    diagnosis_dict = {'M':0 , 'B':1}
    data.diagnosis = data.diagnosis.map(diagnosis_dict)
    
    X = data.drop('diagnosis', axis=1)
    y = data.diagnosis
    
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, stratify=y)
    
    x_train /= 255
    x_valid /= 255
    nb_classes = 2
    y_train = to_categorical(y_train, nb_classes)
    y_valid = to_categorical(y_valid, nb_classes)
    return x_train, y_train, x_valid, y_valid


def model(x_train, y_train, x_valid, y_valid):
    '''
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Dense(512, input_shape=(30,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0, 1)}}))

    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(100))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))

    model.add(Dense(2))
    model.add(Activation('relu'))

    rms = RMSprop()
    model.compile(loss='binary_crossentropy', 
                optimizer={{choice(['adam', 'rmsprop', 'sgd'])}}, 
                metrics=['accuracy'])



    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=30,
              verbose=2,
              validation_data=(x_valid, y_valid))
    score, acc = model.evaluate(x_valid, y_valid, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    for trial in trials:
        print(trial) 

    x_train, y_train, x_valid, y_valid = data()


    print("Evalutation of best performing model:")
    print(best_model.evaluate(x_valid, y_valid))

