import numpy as np
import pandas as pd
size = 100
t_max = 5
t = np.linspace(0, t_max, size)
mu_fr, sigma_fr = 0, 5
fr = np.random.normal(mu_fr, sigma_fr, size)
start_st, end_st = 0.01, 5
st = np.logspace(start_st, end_st, size, endpoint = True)

# the function that we generate the data with
def f(t, st, fr):
    return st**2 * (fr * ((1- t)/st - np.exp(-t/st)))
data = []
for st_ in st:
    for fr_ in fr:
        example = list(f(t, st_, fr_))
        t_pred = np.random.uniform(0, t_max)
        pred = f(t_pred,st_,fr_)
        example.append(fr_)
        example.append(st_)
        example.append(t_pred)
        example.append(pred)
        data.append(example)
data = np.array(data)
colummns = [str(i) for i in range(size)]
colummns.append("fr")
colummns.append("st")
colummns.append("t_pred")
colummns.append("pred")
df = pd.DataFrame(data,columns=colummns)
df.to_csv("data.csv")