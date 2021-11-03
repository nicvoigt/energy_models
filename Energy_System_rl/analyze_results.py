import pandas as pd
from Energy_Models.utils.paths import paths
import os
import matplotlib.pyplot as plt


df = pd.read_csv(os.path.join(paths.output_rl ,"test.csv"))
print(df.head())
plt.plot(df["0"])
plt.show()