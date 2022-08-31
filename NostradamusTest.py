# import the necessary packages

from NostradamusUtils import *
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import copy
import os

global models
models = []
########################################################################################################
########################################################################################################
def SNN(data,networks_number):
    global models

    lista_pred = []

    for i in range(100):##########################
        model = models[i]
        pred = model.predict([data, ])[0]
        lista_pred.append(pred)

    sum = 0
    for p in lista_pred:
        sum += p
    avg_pred = sum / len(lista_pred)

    if avg_pred < 0.5:
        label = "down"
    else:
        label = "up"

    return label, avg_pred, lista_pred
########################################################################################################
def test(dataset_up, dataset_down, up_values, down_values,networks_number):

    global models

    dataset = dataset_down
    dataset.extend(dataset_up)

    values = down_values
    values.extend(up_values)

    up = 0
    down = 0
    up_error = 0
    down_error = 0
    uncertain = 0
    uncertain_up = 0
    uncertain_down = 0
    effective_up = 0
    effective_down = 0
    up_correct = 0
    down_correct = 0
    correct_up_values = []
    correct_down_values = []
    error_up_values = []
    error_down_values = []
    count = 0
    #####################################################

    for i in range(len(dataset)):

        data = dataset[i]

        expectedClass = data[-1]

        if expectedClass == 1. :
            expectedClass = "up"
            effective_up += 1

        else:
            expectedClass = "down"
            effective_down += 1
        count += 1

        result = SNN(data[:-1],networks_number)

        avg_pred = result[1]
        lista_pred = result[2]

        if avg_pred >= 0.55:#####################
            label = "up"
        elif avg_pred < 0.45:#######################
            label = "down"
        else:
            label = "unknown"

        result = f"{label}: {avg_pred}"
        print(result, lista_pred)
        print(label + " vs " + expectedClass)

        if label == "up":
            up += 1
        elif label == "unknown":
            uncertain += 1
        else:
            down += 1

        if label != expectedClass:

            if label == "up":
                up_error += 1
                error_up_values.append(values[i+1])
            elif label == "unknown" and expectedClass == "up":
                uncertain_up += 1
            elif label == "unknown" and expectedClass == "down":
                uncertain_down += 1
            else:
                down_error += 1
                error_down_values.append(values[i+1])

        if label == expectedClass:

            if label == "up":
                up_correct += 1
                correct_up_values.append(values[i+1])
            else:
                down_correct += 1
                correct_down_values.append(values[i+1])

        if (down_error + down_correct) != 0 and (up_error + up_correct) != 0:
            print(f"-down ratio: {down_correct / (down_error + down_correct)} -up ratio: {up_correct / (up_error + up_correct)} -count: {count}")

        print(f"-down frequency: {(down_correct + down_error)/ count} -up frequency: {(up_correct + up_error) / count}")

        avarage_correct_up_values = np.mean(correct_up_values)
        avarage_correct_down_values = np.mean(correct_down_values)
        avarage_error_up_values = np.mean(error_up_values)
        avarage_error_down_values = np.mean(error_down_values)

        print(f"-avarage correct up values: {avarage_correct_up_values} -avarage correct down values: {avarage_correct_down_values}")
        print(f"-avarage error up values: {avarage_error_up_values} -avarage error down values: {avarage_error_down_values}")

    return [down_correct / (down_error + down_correct), up_correct / (up_error + up_correct),
            (down_correct + down_error)/ count, (up_correct + up_error) / count, avarage_correct_up_values,
            avarage_error_up_values, avarage_correct_down_values, avarage_error_down_values, count]
########################################################################################################
########################################################################################################
if __name__ == "__main__":
    lista_results = []
    file_path= "EURUSD_4_1997-01-01_2022-01-02.csv"
    modelPaths = os.listdir("models")
    networks = len(modelPaths)
    cnt = 0
    for modelPath in modelPaths:
        cnt += 1
        print(f"[INFO] {cnt}/{networks} model loaded")
        model = load_model(f"models\\{modelPath}")
        models.append(model)


    dataset_up, dataset_down, up_values, down_values  = TestDatasetFromCsv(file_path, 23000)

    for i in range(1):############################################
        print(f"[INFO] testing {i} networks...")
        result = test(copy.deepcopy(dataset_up), copy.deepcopy(dataset_down), copy.deepcopy(up_values), copy.deepcopy(down_values),i+1)
        lista_results.append(result)

    # print result on file
    file = open("results.txt", "w")

    for result in lista_results:
        file.write(str(result) + "\n")
    file.close()

    # plot results
    dwr_history = []
    upr_history = []
    for i in range(networks):
        dwr_history.append(lista_results[i][1])
        upr_history.append(lista_results[i][2])

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, networks + 1), dwr_history, label="down prediction ratio")
    plt.plot(np.arange(1, networks + 1), upr_history, label="up prediction ratio")

    plt.title("Results on test ")
    plt.xlabel("Number of Neural Networks")
    plt.ylabel("Rate")
    plt.legend(loc="lower left")
    plt.savefig("plot.jpg")
    plt.close()


