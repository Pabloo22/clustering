import pandas as pd
from scipy.stats import shapiro
from sklearn.preprocessing import PowerTransformer
import sys

data = pd.read_csv(sys.argv[1])
data.drop(columns=['id'], inplace=True)

# Check for normality
for col in data.columns:
    stat, p = shapiro(data[col])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        data[col] = PowerTransformer().fit_transform(data[col].values.reshape(-1, 1))

data.to_csv('data/data_powertransform.csv', index=False)
