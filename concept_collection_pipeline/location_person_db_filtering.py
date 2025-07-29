import pandas as pd 
import numpy as np

# used for extracting names from the notable persons database 
name_df = pd.read_csv('data/cross-verified-database.csv', encoding="utf-8", sep=",", encoding_errors="replace")
name_db = name_df["name"].apply(lambda x: x.lower().replace("_", " ") if isinstance(x, str) else x)
name_db.to_csv("data/name_db.csv", index=False)

# used for loading location databases from the notable persons database 
GeoNames_DB = pd.read_csv('data/allCountries.txt', sep="\t", low_memory=False).iloc[:,1].drop_duplicates()
df2 = pd.read_csv('data/worldcities.csv', low_memory=False)

# Concatenate the locations from both datasets, clean the data and save the file
GeoNames_DB = pd.concat([GeoNames_DB.iloc[:, 1], df2.iloc[:, 1]]).drop_duplicates()
GeoNames_DB = GeoNames_DB.apply(lambda x: str(x).lower() if pd.notna(x) and str(x).replace(' ', '').isalpha() else np.nan)
GeoNames_DB = GeoNames_DB.dropna()
GeoNames_DB.to_csv("data/GeoNames_DB.csv", index=False)