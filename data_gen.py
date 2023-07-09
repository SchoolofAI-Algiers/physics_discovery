import numpy as np
import json
import os
import uuid
size = 100
mu_t, sigma_t = 0, 1
t = np.random.normal(mu_t, sigma_t, size)
mu_fr, sigma_fr = 0, 5
fr = np.random.normal(mu_fr, sigma_fr, size)
mu_st, sigma_st = 0.01, 5
st = np.random.lognormal(mu_st, sigma_st, size)
# the function that we generate the data with
def f(t, st, fr):
    return st**2 * (fr * ((1- t)/st - np.exp(-t/st)))
# create a folder to store all the timeseries
if os.path.exists("data") == False:
    os.mkdir("data")
data = []
for st_ in st:
    for fr_ in fr:
        path = f"data/{uuid.uuid4()}.npy"
        np.save(path, f(t, st_, fr_))
        data.append(
            {
                "time_serie" : path,
                "st" : st_,
                "fr" : fr_
            }
        )
print("data size : ", len(data))
# save it as a json 
with open("data.json", "w") as json_file:
    json.dump({"data" : data}, json_file)