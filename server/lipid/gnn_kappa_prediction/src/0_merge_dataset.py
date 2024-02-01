import pandas as pd

data1=pd.read_excel('../data/2-component-results-formatted.xlsx')
data2= pd.read_excel('../data/Selected_Bending-modulus-Features_Model.xls')

# Convert to pandas DataFrames
df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# Calculate the mean of the specific column
mean_value = df2['Kappa_rsf'].mean()

# Fill missing (NaN) values in the column with the mean
df2['Kappa_rsf'].fillna(mean_value, inplace=True)


df2.drop(columns=['Salt, M'], inplace=True)

# Rename columns in df2 to match df1
df2.rename(columns={
    "Lipid composition (molar)": "Composition",
    # "Salt, M": "Salt Concentration",
    "N_lipids/layer": "N Lipids/Layer",
    "Temperature, K": "Temperature (K)",
    "Memb_thickness": "Avg Membrane Thickness",
    "kappa, kT (q^-4)": "Kappa (q^-4)",
    "Kappa  BW-DCF": "Kappa (BW-DCF)",
    "Kappa_rsf": "Kappa (RSF)"
}, inplace=True)



# Reorder columns in df2 to match df1
df1 = df1[df2.columns]


# print(df1)
# Concatenate the datasets
merged_df = pd.concat([df1, df2], ignore_index=True)

def composition_sort_key(composition):
    if ',' in str(composition):
        return 1
    else:
        return 0

merged_df.dropna(axis=1, how='all', inplace=True)

# Apply sorting
merged_df['sort_key'] = merged_df['Composition'].apply(composition_sort_key)
merged_df.sort_values(by='sort_key', inplace=True)
merged_df.drop(columns=['sort_key'], inplace=True)  # Remove the temporary sorting column

# Save the merged and sorted DataFrame
merged_df.to_csv('../data/Merged_and_Sorted_Df.csv', index=False)