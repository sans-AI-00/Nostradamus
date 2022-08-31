##################################################################################################################
##################################################################################################################
import csv
import numpy as np
import copy
##################################################################################################################
##################################################################################################################
def normalization(sample, vol_periods, max_rsi, min_rsi, max_vol,min_vol):

    sample[0] = (sample[0] - min_rsi)/ (max_rsi - min_rsi)
    for j in range(1, len(vol_periods)+1):
        sample[-j] = (sample[-j] - min_vol[-j])/ (max_vol[-j] - min_vol[-j])
    return sample


##################################################################################################################
def straddling(MA):

    straddling_vector = []

    for k in range(len(MA[0])):
        straddling_subvector = []

        for i in range(len(MA)-1):

            for j in range(i+1, len(MA)):

                if MA[i][k] >= MA[j][k]:
                    straddling_subvector.append(0)

                else:
                    straddling_subvector.append(1)

        straddling_vector.append(straddling_subvector)

    return straddling_vector
##################################################################################################################
def trasp(matrix):

    matrix_result = []

    for j in range(len(matrix[0])):
        vector_result = []
        for i in range(len(matrix)):
            vector_result.append(matrix[i][j])
        matrix_result.append(vector_result)

    return matrix_result
##################################################################################################################
def ma(values, period, type = "sma"):

     values = copy.deepcopy(values)
     ma_vector = []

     if type == "sma":

         for i in range(len(values)):

             if i+1 < period:
                 ma_vector.append(0)

             elif (i + 1) >= period:
                 sum = 0

                 for j in range(period):
                     sum += values[i - j]

                 ma_vector.append(sum/period)

     return ma_vector
##################################################################################################################
def rsi(Asset,period=14):

    close_values = copy.deepcopy(Asset.close)
    open_values = copy.deepcopy(Asset.open)

    rsi_vector = []

    green_red_list = []

    for i in range(len(close_values)):

        if (close_values[i] - open_values[i]) >= 0:
            close_values[i] = close_values[i] - open_values[i]
            green_red_list.append(0)

        elif (close_values[i] - open_values[i]) < 0:
            close_values[i] = open_values[i] - close_values[i]
            green_red_list.append(1)

    for i in range(len(close_values)):
        green_data = len(green_red_list[:i+1]) - sum(green_red_list[:i+1])
        red_data = sum(green_red_list[:i + 1])

        if green_data < period or red_data < period:
            rsi_vector.append(0)

        else:
            sum_green = []
            sum_red = []
            green_count = 0
            red_count = 0

            for j in range(i+1):

                if green_red_list[i - j] == 0 and green_count < period:
                    sum_green.append(close_values[i - j])
                    green_count += 1

                elif green_red_list[i - j] == 1 and red_count < period:
                    sum_red.append(close_values[i - j])
                    red_count += 1

                if green_count == period and red_count == period:
                    break

            avarage_green_close = sum(sum_green)/len(sum_green)
            avarage_red_close = sum(sum_red) / len(sum_red)

            rs = avarage_green_close/avarage_red_close
            rsi = 100 - (100 / (1 + rs))
            rsi_vector.append(rsi)

    return rsi_vector
##################################################################################################################
def volatility(values,period):

    values = copy.deepcopy(values)
    volatility_vector = []

    for i in range(len(values)):

        if i + 1 < period:
            volatility_vector.append(0)

        elif (i + 1) >= period:

            volatility_vector.append(np.std(values[i-period+1:i+1])/np.mean(values[i-period+1:i+1]))

    return volatility_vector
##################################################################################################################
def DatasetFromCsv(file_path):
    with open(file_path, newline="", encoding="ISO-8859-1") as filecsv:

        lettore = csv.reader(filecsv, delimiter=",")
        dataset = []
        cnt = 0

        for row in lettore:
            row[0] = cnt
            del row[1]
            dataset.append(row)
            cnt += 1

    #dataset = dataset[1:]
    dataset = trasp(dataset)

    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j] = float(dataset[i][j])

    return dataset
##################################################################################################################
def upANDdown_values(close, open):

    up_values = []
    down_values = []

    for i in range(len(close)):
        value = close[i] - open[i]
        avarage_value = (close[i] + open[i]) / 2

        if value >= 0:
            up_values.append(value / avarage_value)
        else:
            down_values.append(value / avarage_value)

    return up_values, down_values
##################################################################################################################
def TestDatasetFromCsv(file_path,start_value):

    dataset = DatasetFromCsv(file_path)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][start_value:]
    EURUSD = Asset.from_dataset("EURUSD", dataset)

    dataset_up = []
    dataset_down = []

    MA_periods = [1, 3, 5, 7, 10, 15, 20, 35, 50, 75, 100, 125, 150, 175, 200]
    Volatility_periods = [5, 7, 10, 20, 50, 100, 200]

    features = []
    MA_price = []
    MA_volume = []
    Volatility = []

    RSI = (EURUSD.RSI())

    for period in MA_periods:
        MA_price.append(EURUSD.MA_price(period))
        MA_volume.append(EURUSD.MA_volume(period))

    MA_str_price = straddling(copy.deepcopy(MA_price))
    MA_str_volume = straddling(copy.deepcopy(MA_volume))

    for period in Volatility_periods:
        Volatility.append(EURUSD.Volatility(period))

    max_rsi = max(copy.deepcopy(RSI))
    min_rsi = min(copy.deepcopy(RSI))

    max_vol = copy.deepcopy(Volatility)
    min_vol = copy.deepcopy(Volatility)

    for i in range(len(max_vol)):
        max_vol[i] = max(copy.deepcopy(Volatility)[i])
        min_vol[i] = min(copy.deepcopy(Volatility)[i])

    features.append(RSI)
    features.extend(Volatility)

    for i in range(365, len(EURUSD.close) - 2):
        sample = [RSI[i]]
        sample.extend(MA_str_price[i])
        sample.extend(MA_str_volume[i])
        for j in range(len(Volatility)):
            sample.append(Volatility[j][i])

        sample = normalization(copy.deepcopy(sample), Volatility_periods, max_rsi, min_rsi, max_vol, min_vol)

        sample.append(EURUSD.Trend(i + 1))

        if EURUSD.Trend(i + 1) == 1.:
            dataset_up.append(sample)

        else:

            dataset_down.append(sample)

    up_values = EURUSD.up_values()
    down_values = EURUSD.down_values()

    len_list = [len(dataset_up),len(dataset_down)]

    min_len = min(len_list)

    dataset_up = dataset_up[:min_len]
    up_values = up_values[:min_len]
    dataset_down = dataset_down[:min_len]
    down_values = down_values[:min_len]

    return dataset_up, dataset_down, up_values, down_values
##################################################################################################################
def TrainingDatasetFromCsv(file_path,stop_value):

    dataset = DatasetFromCsv(file_path)
    for i in range(len(dataset)):
        dataset[i] = dataset[i][:stop_value]
    EURUSD = Asset.from_dataset("EURUSD", dataset)

    dataset_up = []
    dataset_down = []

    MA_periods = [1, 3, 5, 7, 10, 15, 20, 35, 50, 75, 100, 125, 150, 175, 200]
    Volatility_periods = [5, 7, 10, 20, 50, 100, 200]

    features = []
    MA_price = []
    MA_volume = []
    Volatility = []

    RSI = (EURUSD.RSI())

    for period in MA_periods:
        MA_price.append(EURUSD.MA_price(period))
        MA_volume.append(EURUSD.MA_volume(period))

    MA_str_price = straddling(copy.deepcopy(MA_price))
    MA_str_volume = straddling(copy.deepcopy(MA_volume))

    for period in Volatility_periods:
        Volatility.append(EURUSD.Volatility(period))

    max_rsi = max(copy.deepcopy(RSI))
    min_rsi = min(copy.deepcopy(RSI))

    max_vol = copy.deepcopy(Volatility)
    min_vol = copy.deepcopy(Volatility)

    for i in range(len(max_vol)):
        max_vol[i] = max(copy.deepcopy(Volatility)[i])
        min_vol[i] = min(copy.deepcopy(Volatility)[i])

    features.append(RSI)
    features.extend(Volatility)

    for i in range(365, len(EURUSD.close) - 2):
        sample = [RSI[i]]
        sample.extend(MA_str_price[i])
        sample.extend(MA_str_volume[i])
        for j in range(len(Volatility)):
            sample.append(Volatility[j][i])

        sample = normalization(copy.deepcopy(sample), Volatility_periods, max_rsi, min_rsi, max_vol, min_vol)

        sample.append(EURUSD.Trend(i + 1))

        if EURUSD.Trend(i + 1) == 1.:
            dataset_up.append(sample)

        else:

            dataset_down.append(sample)

    return dataset_up, dataset_down
##################################################################################################################
##################################################################################################################
class Asset:

    def __init__(self, symbol, count, open, high, low, close, volume):

        self.symbol = symbol
        self.count = count
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @classmethod
    def from_dataset(cls, symbol, dataset, *args):

        count, open, high, low, close, volume = dataset


        return cls(symbol, count, open, high, low, close, volume, *args)

    def show(self):

        output_string = f" count: {self.count}: \n open: {self.open} \n high: {self.high} \n " \
                        f"low: {self.low} \n close: {self.close} \n volume: {self.volume}"

        return output_string

    def MA_price(self, period, type="sma"):

        return ma(self.close, period, type)

    def MA_volume(self, period, type="sma"):

        return ma(self.volume, period, type)

    def RSI(self, period=14):

        return rsi(self, period)

    def Volatility(self, period):

        return volatility(self.close, period)

    def up_values(self):

        return upANDdown_values(self.close,self.open)[0]

    def down_values(self):

        return upANDdown_values(self.close, self.open)[1]

    def Trend(self, index):

        if self.close[index] >= self.open[index]:

            return 1.

        else:

            return 0.
##################################################################################################################
##################################################################################################################

