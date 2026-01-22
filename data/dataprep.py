# Load packages
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

# Load raw data
df = pd.read_csv('/content/final_sorted.csv')

# Data imputation
imputer = KNNImputer(n_neighbors=5, copy=False)

for year in np.unique(df['year']):
    X = df.loc[df['year'] == year, ['electricity_aid', 'electricity_access', 'rural_electricity_access', 'urban_electricity_access',
                                     'fdi', 'gdp', 'gdp_per_capita', 'inflation', 'life_expectancy',
                                     'popu_growth', 'popu_total', 'rural_popu_growth', 'urban_popu_growth', 'unemployment']]
    X = imputer.fit_transform(X)
    df.loc[df['year'] == year, ['electricity_aid', 'electricity_access', 'rural_electricity_access', 'urban_electricity_access',
                                'fdi', 'gdp', 'gdp_per_capita', 'inflation', 'life_expectancy',
                                'popu_growth', 'popu_total', 'rural_popu_growth', 'urban_popu_growth', 'unemployment']] = X

# Add lag for electricity access
df['electricity_access_lag'] = df.groupby('country')['electricity_access'].shift(1)

# Calculate electricity access improvement
df['electricity_access_improvement'] = (df['electricity_access'] - df['electricity_access_lag']) / df['electricity_access_lag']

# Drop rows with missing values
df = df.dropna(axis=0)

df.to_csv('electricity_data.csv', index=False)
