import pandas as pd

# Read the CSV file
electricity_aid = pd.read_csv("/content/elect_aid.csv") # header=None

# Select specific columns
electricity_aid = electricity_aid.iloc[:, 0:5]

# Rename columns
electricity_aid.columns = ['country', '2019', '2020', '2021', '2022']

# Replace '..' with NA
electricity_aid.replace('..', pd.NA, inplace=True)

# Count missing values in each row
no_miss = electricity_aid.isna().sum(axis=1)

# Remove rows with more than 2 missing values
electricity_aid = electricity_aid[no_miss <= 2]

# Reshape the data frame from wide to long format
electricity_aid = electricity_aid.melt(id_vars=['country'], var_name='year', value_name='electricity_aid')

# Convert columns to numeric
electricity_aid['year'] = pd.to_numeric(electricity_aid['year'])
electricity_aid['electricity_aid'] = pd.to_numeric(electricity_aid['electricity_aid'])

# Replace missing values in the 'electricity_aid' column with the string "NA"
electricity_aid['electricity_aid'] = electricity_aid['electricity_aid'].fillna('NA')

# Display the resulting data frame
print(electricity_aid)

# Save the resulting data frame to a CSV file
#electricity_aid.to_csv('cleaned_electricity_aid.csv', index=False)

# Read the CSV files
elect_access_popu = pd.read_csv("/content/elect_access_popu.csv", skiprows=4)
elect_access_rural_popu = pd.read_csv("/content/elect_access_rural_popu.csv", skiprows=4)
elect_access_urban_popu = pd.read_csv("/content/elect_access_urban_popu.csv", skiprows=4)
fdi = pd.read_csv("/content/fdi.csv", skiprows=4)
gdp = pd.read_csv("/content/gdp_growth_annual.csv", skiprows=4)
gdp_per_capita = pd.read_csv("/content/gdp_per_capita_ppp.csv", skiprows=4)
inflation = pd.read_csv("/content/inflation.csv", skiprows=4)
life_expectancy = pd.read_csv("/content/life_expectancy_birth.csv", skiprows=4)
popu_growth = pd.read_csv("/content/popu_growth.csv", skiprows=4)
popu_total = pd.read_csv("/content/popu_total.csv", skiprows=4)
rural_popu_growth = pd.read_csv("/content/rural_popu_growth.csv", skiprows=4)
urban_popu_growth = pd.read_csv("/content/urban_popu_growth.csv", skiprows=4)
unemployment = pd.read_csv("/content/unemployment.csv", skiprows=4)

# Combine the data frames
covariates = pd.concat([
    elect_access_popu, elect_access_rural_popu, elect_access_urban_popu, fdi, gdp, gdp_per_capita,
    inflation, life_expectancy, popu_growth, popu_total, rural_popu_growth, urban_popu_growth, unemployment
    ], ignore_index=True)

# Select specific columns
covariates = covariates.iloc[:, [0, 1, 2, 63, 64, 65, 66]]

# Reshape the data frame from wide to long format
covariates = covariates.melt(id_vars=['Country Name', 'Country Code', 'Indicator Name'], var_name='year', value_name='value')

# Spread the data frame from long to wide format
covariates = covariates.pivot(index=['Country Name', 'Country Code', 'year'], columns='Indicator Name', values='value').reset_index()

# Rename columns
covariates.columns = ['country', 'id', 'year', 'electricity_access', 'rural_electricity_access', 'urban_electricity_access',
                      'fdi', 'gdp', 'gdp_per_capita', 'inflation', 'life_expectancy', 'popu_growth', 'popu_total', 'rural_popu_growth',
                      'urban_popu_growth', 'unemployment']

# Assign years
covariates['year'] = [2019, 2020, 2021, 2022] * len(covariates['country'].unique())

# Scale values
covariates['fdi'] = covariates['fdi'] / 1000000
covariates['gdp_per_capita'] = covariates['gdp_per_capita'] / 1000

covariates['popu_total'] = (
    covariates['popu_total']
    .str.replace(',', '', regex=False)  # Remove commas
    .astype(float)  # Convert to numeric
)

covariates['popu_total'] = covariates['popu_total'] / 1000000

# Display the resulting data frame
print(covariates)
#covariates.to_csv("covariates_output.csv", index=False)

# Extract unique country names
ctry_names1 = electricity_aid['country'].unique()
ctry_names2 = covariates['country'].unique()

# Find mismatched country names
mismatched_countries = [country for country in ctry_names1 if country not in ctry_names2]
print(f"Mismatched countries: {len(mismatched_countries)}")

# Rename countries in hiv_aid
electricity_aid['country'] = electricity_aid['country'].replace({
    "China (People's Republic of)": "China",
    "Congo": "Congo, Rep.",
    "Democratic Republic of the Congo": "Congo, Dem. Rep.",
    "Democratic People's Republic of Korea": "Korea, Dem. People's Rep.",
    "Egypt": "Egypt, Arab Rep.",
    "Gambia": "Gambia, The",
    "Iran": "Iran, Islamic Rep.",
    "Kyrgyzstan": "Kyrgyz Republic",
    "Lao People's Democratic Republic": "Lao PDR",
    "Viet Nam": "Vietnam",
    "Venezuela": "Venezuela, RB",
    "West Bank and Gaza Strip": "West Bank and Gaza",
    "Yemen": "Yemen, Rep."
})

# Update unique country names
ctry_names1 = electricity_aid['country'].unique()
mismatched_countries = [country for country in ctry_names1 if country not in ctry_names2]
print(f"Mismatched countries now: {len(mismatched_countries)}")

# Merge data frames
aid_covariates = pd.merge(electricity_aid, covariates, on=['country', 'year'])

# Sort the DataFrame by 'country' first, then by 'year'
final_sorted = aid_covariates.sort_values(by=['country', 'year'])

# Display the resulting data frame
print(final_sorted)

# Save the resulting data frame to a CSV file
final_sorted.to_csv('final_sorted.csv', index=False)
