from NostradamusUtils import *

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout

import random

print(tf.__version__)
##################################################################################################################
##################################################################################################################
def NN_training(dataset_up, dataset_down, network_index):

    dataset_train = [dataset_up.pop(random.randint(0,len(dataset_up)-1))]
    dataset_train.append(dataset_down.pop(random.randint(0,len(dataset_down)-1)))
    for i in range(7499):
        dataset_train.append(dataset_up.pop(random.randint(0,len(dataset_up)-1)))
        dataset_train.append(dataset_down.pop(random.randint(0, len(dataset_down) - 1)))

    dataset_test = [dataset_up.pop(random.randint(0,len(dataset_up)-1))]
    dataset_test.append(dataset_down.pop(random.randint(0,len(dataset_down)-1)))
    for i in range(2499):
        dataset_test.append(dataset_up.pop(random.randint(0, len(dataset_up)-1)))
        dataset_test.append(dataset_down.pop(random.randint(0, len(dataset_down) - 1)))

    print(len(dataset_train),len(dataset_test))

    ##################################################################################################################
    ##################################################################################################################
    Xtrain = []
    ytrain = []
    for sample in dataset_train:
        ytrain.append(sample[-1])

    for i in range(len(dataset_train)):
        Xtrain.append(dataset_train[i][:len(dataset_train[i]) - 1])

    Xtest = []
    ytest = []
    for sample in dataset_test:
        ytest.append(sample[-1])

    for i in range(len(dataset_test)):
        Xtest.append(dataset_test[i][:len(dataset_test[i]) - 1])

    Xtrain = np.array(Xtrain)
    Xtest = np.array(Xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)

    width = len(Xtrain[0])
    height = 1
    channels = 1

    Xtrain = Xtrain.reshape(len(Xtrain), width, )
    Xtest = Xtest.reshape(len(Xtest), width, )
    ##################################################################################################################
    ##################################################################################################################

    model = Sequential()
    model.add(Dense(1000, activation='sigmoid', input_shape=(width,)))
    #model.add(Dropout(0.1))
    model.add(Dense(200, activation='sigmoid'))#softmax/sigmoid
    model.add(Dropout(0.1))
    model.add(Dense(100, activation='sigmoid'))#softmax/sigmoid
    #model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
              optimizer='Adamax',
              metrics=['accuracy'])

    epochs = 10
    batch_size = 100
    history = model.fit(Xtrain, ytrain, batch_size, epochs, verbose=1,
                        validation_data=(Xtest, ytest))

    score = model.evaluate(Xtest, ytest, verbose=0)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    name = f"models\\model_{network_index}.h5"
    model.save(name)

    return None
##################################################################################################################
##################################################################################################################
if __name__=='__main__':
    file_path = "EURUSD_4_1997-01-01_2022-01-02.csv"
    dataset_up, dataset_down = TrainingDatasetFromCsv(file_path,stop_value=23000)
    for i in range(24,1000):
        print(f"[INFO] addestramento rete numero {i}...")
        NN_training(copy.deepcopy(dataset_up), copy.deepcopy(dataset_down), i)
        print("[INFO] addestramento terminato con successo")