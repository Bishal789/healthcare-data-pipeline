import pandas as pd

# Load the raw CSV exactly as downloaded — no cleaning yet
df = pd.read_csv("data/raw/diabetic_data.csv")

print("Shape:", df.shape)
print("\nColumn data types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
#print("\nMissing values per column (top 10):\n", df.isnull().sum().sort_values(ascending=False).head(10))

# Replace it so pandas can actually recognize what's missing
df_check = df.replace("?", pd.NA)

print("\nMissing values per column (top 15), after fixing '?':\n",
      df_check.isnull().sum().sort_values(ascending=False).head(15))

print("\nPercent missing (top 15):\n",
      (df_check.isnull().mean() * 100).sort_values(ascending=False).head(15))

print()