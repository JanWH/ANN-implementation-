import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

df = pd.read_excel(r"FEHDataStudent.xlsx")

raw_data = df[["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR"]].to_numpy()#write selected rows to numpy array

index_flood = df[["Index flood"]].to_numpy()#same for results

def check_erroneous(raw_array, raw_results):#removes NaN cells and negative numbers
    i = 0
    index_list = []
    for x in raw_array:
        for y in x:
            if isinstance(y, (float, int)) == False or np.isnan(y) or y<0:
                index_list.append(i)
        i+=1
    new_raw_array = np.delete(raw_array, index_list, 0)
    new_raw_results = np.delete(raw_results, index_list)
    return new_raw_array, new_raw_results

def remove_outliers(raw_array, raw_results):
    three_standard_dev = 3*np.std(raw_array, axis=0, dtype=np.float32)
    mean = np.mean(raw_array, axis = 0, dtype=np.float32)
    upper = mean+three_standard_dev
    lower = mean-three_standard_dev
    index_list = []
    j = 0
    for x in raw_array:
        for i in range(7):
            if x[i] > upper[i] or x[i] < lower[i]:
                index_list.append(j)
        j+=1
    new_raw_array = np.delete(raw_array, index_list, 0)
    new_raw_results = np.delete(raw_results, index_list)
    return new_raw_array, new_raw_results

def standardise(raw_array, raw_results):
    arraymax = np.amax(raw_array, axis=0)
    resmax = np.max(raw_results)
    data_mands = np.array([np.std(raw_array, axis=0, dtype=np.float32),np.mean(raw_array, axis = 0, dtype=np.float32)])
    index_flood_mands = np.array([np.mean(raw_results), np.std(raw_results)])
    mands = np.hstack([data_mands, index_flood_mands[:,None]])
    raw_array = raw_array.T
    for i in range(len(raw_array)):
        if arraymax[i]>1:
            raw_array[i] = (raw_array[i]-np.mean(raw_array[i], keepdims=True))/np.std(raw_array[i])
    if resmax>1:
        raw_results = (raw_results-np.mean(raw_results, keepdims=True))/np.std(raw_results)
    raw_array = raw_array.T
    return raw_array, raw_results, mands

def normalise(raw_array, raw_results):
    arraymax = np.amax(raw_array, axis=0)
    resmax = np.max(raw_results)
    arraymin = np.amin(raw_array, axis=0)
    resmin = np.min(raw_results)
    minmax = np.array([np.append(arraymax, resmax), np.append(arraymin, resmin)])
    raw_array = raw_array.T
    for i in range(len(raw_array)):
        if arraymax[i]>1:
            for j in range(len(raw_array[i])):
                raw_array[i][j] = 0.8*((raw_array[i][j]-arraymin[i])/(arraymax[i]-arraymin[i]))+0.1
    if resmax>1:
        for i in range(len(raw_results)):
            raw_results[i] = 0.8*((raw_results[i]-resmin)/(resmax-resmin))+0.1
    raw_array = raw_array.T
    return raw_array, raw_results, minmax


raw_data, index_flood = check_erroneous(raw_data, index_flood)
raw_data, index_flood = remove_outliers(raw_data, index_flood)
#print(raw_data[0])
choose = input("to normalise type n, to standardise type s")
if choose == "n":
    raw_data, index_flood, minmax = normalise(raw_data, index_flood)
    dfminmax = pd.DataFrame(data = minmax,
                       columns = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","INDEXFLOOD"],
                       index = ["Max","Min"])
    dfminmax.to_excel('minmax.xlsx')

    dataout = np.hstack([raw_data, index_flood[:,None]])

    dfclean = pd.DataFrame(data = dataout,
                        columns = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","INDEXFLOOD"])
    dfclean.to_excel('normalised.xlsx', index=False)
    splitter = dfclean.sample(frac=1, replace=False)
    splitter[:math.ceil(len(splitter)*0.6)].to_excel('training.xlsx', index=False)
    splitter[math.ceil(len(splitter)*0.6)+1:math.ceil(len(splitter)*0.8)].to_excel('verify.xlsx', index=False)
    splitter[math.ceil(len(splitter)*0.8)+1:].to_excel('testing.xlsx', index=False)

else:
    raw_data, index_flood, mands = standardise(raw_data, index_flood)
    dfmands = pd.DataFrame(data = mands,
                       columns = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","INDEXFLOOD"],
                       index = ["Mean","StanDev"])
    dfmands.to_excel('mands.xlsx')

    dataout = np.hstack([raw_data, index_flood[:,None]])

    dfclean = pd.DataFrame(data = dataout,
                        columns = ["AREA","BFIHOST","FARL","FPEXT","LDP","PROPWET","RMED-1D","SAAR","INDEXFLOOD"])
    dfclean.to_excel('standardised.xlsx')


