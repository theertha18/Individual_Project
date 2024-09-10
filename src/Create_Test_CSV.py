import pandas as pd

import os

# Load the CSV file
df = pd.read_csv('D:\\individualProject\\Dataset\\adc_plainwood_1m.csv')

# Delete 999 rows
df = df.iloc[998:]

# Save the filtered DataFrame back to CSV
output_file_path = 'D:\\individualProject\\Dataset\\test_adc_steel_1m.csv'
df.to_csv(output_file_path, index=False)

# Set permissions for the file
os.chmod(output_file_path, 0o644)  # This sets the file permissions to 644 (read/write for owner, read for others)
