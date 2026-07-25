# ============================================================
# Adult Income Dataset - Data Preprocessing
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------

df = pd.read_csv("adult.csv")

print("=" * 50)
print("Original Dataset Shape :", df.shape)

# ------------------------------------------------------------
# Remove Duplicate Rows
# ------------------------------------------------------------

duplicate_count = df.duplicated().sum()
print(f"\nDuplicate Rows Found : {duplicate_count}")

df.drop_duplicates(inplace=True)

print("Dataset Shape After Removing Duplicates :", df.shape)

# ------------------------------------------------------------
# Replace '?' with Missing Values
# ------------------------------------------------------------

df.replace(["?", " ?"], np.nan, inplace=True)

# ------------------------------------------------------------
# Remove Leading and Trailing Spaces
# ------------------------------------------------------------

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].str.strip()

# ------------------------------------------------------------
# Fill Missing Values
# ------------------------------------------------------------

# Numerical Columns -> Median
numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Categorical Columns -> Mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# ------------------------------------------------------------
# Encode Target Column
# ------------------------------------------------------------

df["income"] = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})

# ------------------------------------------------------------
# Encode Remaining Categorical Columns
# ------------------------------------------------------------

label_encoders = {}

for column in categorical_columns:

    if column == "income":
        continue

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    label_encoders[column] = encoder

# ------------------------------------------------------------
# Final Validation
# ------------------------------------------------------------

print("\nRemaining Missing Values")
print(df.isnull().sum())

print("\nFinal Dataset Shape :", df.shape)

print("\nData Types")
print(df.dtypes)

# ------------------------------------------------------------
# Save Preprocessed Dataset
# ------------------------------------------------------------

output_file = "adult_preprocessed.csv"

df.to_csv(output_file, index=False)

print(f"\n✅ Preprocessed dataset saved as '{output_file}'")

# ------------------------------------------------------------
# Preview
# ------------------------------------------------------------

print("\nFirst Five Rows")
print(df.head())