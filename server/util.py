import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bedroom,bathroom,reception):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bedroom
    x[2] = bathroom
    x[3] = reception
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations

    with open("server/artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[4:]  # first 4 columns are sqft, bath, bedrooms, bathrooms and  receptions 

    global __model
    if __model is None:
        with open('C:/Users/peter/OneDrive/Documents/london_house_prices/server/artifacts/london_house__prices_model.pickle', "rb") as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")

def get_location_names():
    return __locations

def get_data_columns():
    return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('sw11',1000, 3, 3,3))
    print(get_estimated_price('e1',8000, 2, 2,3))
