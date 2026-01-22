import pandas as pd

# File paths for the four text files
file_paths = {
    '2019': 'e_aid2019.txt',
    '2020': 'e_aid2020.txt',
    '2021': 'e_aid2021.txt',
    '2022': 'e_aid2022.txt'
}

# Read each file and extract the necessary columns
data_frames = {}
for year, path in file_paths.items():
    df = pd.read_csv(path, delimiter='|')
    # Group by RecipientName and sum the USD_Disbursement
    df = df.groupby('RecipientName')['USD_Disbursement'].sum().reset_index()
    # Rename the USD_Disbursement column to reflect the year
    df.rename(columns={'USD_Disbursement': f'{year}_USD_disbursements'}, inplace=True)
    data_frames[year] = df

# Merge all dataframes on RecipientName
combined_df = data_frames['2019']
for year in ['2020', '2021', '2022']:
    combined_df = combined_df.merge(data_frames[year], on='RecipientName', how='outer')

# Save the combined dataframe to a CSV file
combined_df.to_csv('elect_aid.csv', index=False)

print("Combined CSV file has been created as 'combined_disbursements.csv'.")
