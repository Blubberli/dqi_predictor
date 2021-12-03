import pandas as pd
from collections import Counter

data = pd.read_csv("europolis_dqi.csv", sep="\t")
data.dropna()
print(Counter(data.int1))
print(Counter(data.int2))
print(Counter(data.int3))
print(Counter(data.question))
print(Counter(data.resp_gr))
print(Counter(data.jcon))
print(Counter(data.jlev))

