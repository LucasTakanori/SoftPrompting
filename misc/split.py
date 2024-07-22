import os
import json
import random
import pandas as pd

# Load the original JSON file
with open('lt400_fixed.json', 'r') as f:
    data = json.load(f)

# Ensure reproducibility
random.seed(42)

# Shuffle the data
random.shuffle(data)

# Define the sizes for train, dev, and test sets
train_size = 300
dev_size = 50
test_size = 50

# Split the data
train_data = data[:train_size]
dev_data = data[train_size:train_size + dev_size]
test_data = data[train_size + dev_size:train_size + dev_size + test_size]

# # Save the splits to JSON files
# with open('train.json', 'w') as f:
#     json.dump(train_data, f, indent=4)

# with open('dev.json', 'w') as f:
#     json.dump(dev_data, f, indent=4)

# with open('test.json', 'w') as f:
#     json.dump(test_data, f, indent=4)

# Convert the lists of dictionaries to pandas DataFrames
train_df = pd.DataFrame(train_data)
dev_df = pd.DataFrame(dev_data)
test_df = pd.DataFrame(test_data)

# Save the DataFrames to TSV files
train_df.to_csv('train.tsv', sep='\t', index=False)
dev_df.to_csv('dev.tsv', sep='\t', index=False)
test_df.to_csv('test.tsv', sep='\t', index=False)  

print("Data split into train, dev, and test sets and saved to TSV files.")
