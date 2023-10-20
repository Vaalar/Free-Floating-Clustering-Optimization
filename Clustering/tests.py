import pandas as pd

# Create a sample DataFrame
data = {'Group': ['A', 'B', 'A', 'C', 'B', 'C'],
        'Value': [1, 2, 3, 4, 5, 6]}

df = pd.DataFrame(data)

# Group the DataFrame by the 'Group' column
grouped = df.groupby('Group')

# Iterate over each subgroup
for group_name, group_data in grouped:
    print(f"Group Name: {group_name}")
    for _, data in group_data.sor:
        print(data)
        print("\n")
