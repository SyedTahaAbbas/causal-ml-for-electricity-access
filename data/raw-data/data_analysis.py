import pandas as pd

# Load CSV file while skipping the first 4 lines
df = pd.read_csv("//Users/tahazaidi/Desktop/LMU/Thesis2024/Thesis/Shared/mmt-thesis/data/raw-data/elect_access_popu.csv",  skiprows=4) # delimiter=";",

# Calculate key metrics
total_rows = len(df)  # Total rows in the DataFrame
non_null_counts = df.count()  # Non-null values per column
missing_counts = total_rows - non_null_counts  # Missing values per column

# Create summary DataFrame
summary_df = pd.DataFrame({
    "Column Name": df.columns,
    "Non-Null Values": non_null_counts.values,
    "Missing Values": missing_counts.values,
    "Total Values (Rows)": [total_rows] * len(df.columns)  # Total per column (same as rows)
})

# Print results
print("📌 Column-wise Data Summary:\n")
print(summary_df.to_string(index=False))

###########################################

# Load the CSV file
file_path = 'final_sorted.csv'
df = pd.read_csv(file_path)

# Calculate the percentage of missing data for each column
missing_data_percentage = df.isnull().mean() * 100

# Calculate the total percentage of missing data in the entire dataset
total_missing_data = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

# Display the results
print("Percentage of missing data in each column:")
print(missing_data_percentage)
print("\nTotal percentage of missing data in the dataset:")
print(total_missing_data)
