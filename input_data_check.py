import pandas as pd
import matplotlib.pyplot as plt

import random
import numpy as np


df = pd.read_csv(r"C:\Users\nicoj\Kopie_Netcase\1-Start-UP\1-Post Masterarbeit\Programming-Artikel\Pyomo-Serie\df_merged.csv", sep=";")

new =  [df["Prognose"][i] * np.random.uniform(low=0.2,high=1.5, size=(len(df,)))[i] for i in range(len(df))]


plt.plot(new)
plt.plot(df["Reale Last"])
plt.show()

test = np.random.uniform(low=-0.5,high=0.5)
print(test)

df["Prognose"] = new

df.to_csv("df_merged_2.csv")