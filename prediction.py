from fastapi import FastAPI, APIRouter, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import model_from_json
from collections import Counter


def prediction(file_path):
    # Read the contents of the uploaded CSV file
    # Convert the CSV data to a Pandas DataFrame
    df = pd.read_csv(file_path)
    print(df.head())
    column = df.columns
    df = clean_data(df)
    print(df.dtypes)
    test_x = df.iloc[:, 1:]
    test_y = df['label']

    json_file = open('dl_trained_model\\model2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    print("loaded model 0")
    loaded_model = model_from_json(loaded_model_json)
    print("loaded model 1")
    loaded_model.load_weights("dl_trained_model\\model2.h5")
    print("loaded model 2")

    y_pred = loaded_model.predict(test_x)
    print("loaded model 3")
    output_labels = [np.argmax(i) for i in y_pred]
    print("loaded model 4")
    total = len(output_labels)
    freq = dict(Counter(output_labels))
    print(freq)
    pred = list(zip(freq.keys(), freq.values()))
    pred = sorted(pred, reverse=True, key=lambda x: x[1])
    print(pred)
    ans = []
    cnt = 0
    for x in pred:
        print(x[0], x[1])
        ans.append((x[0], (x[1]/total)*100))
        cnt += 1
        if cnt == 3:
            break
    # print(ans)
    # return {"Output": "Prediction executed successfully"}
    return pred


def clean_data(data):
    column = data.columns
    # remove column "Unnamed: 100"
    if 'Unnamed: 100' in data.columns:
        data = data.drop('Unnamed: 100', axis=1)
    print("-----------------------------------------")
    print("Unnamed row removal done\n")

    # remove corrupt rows
    count = 0
    for i in range(len(data)):
        if data['label'][i] == 'label':
            data.drop(i, axis=0, inplace=True)
            count += 1
    data = data.reset_index(drop=True)
    print("-----------------------------------------")
    print("Number of corrupt rows == ", count,
          "[-- corrupt rows removal done--]\n")

    # convert datatype from object to float
    data['label'] = pd.to_numeric(data['label'])

    column = data.columns
    for i in column[1:]:
        data[i] = pd.to_numeric(data[i], errors='coerce')
    print("-----------------------------------------")
    print("Datatype conversion done\n")

    return data
