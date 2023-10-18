import json

# Specify the JSON file path
json_file_path = "data.json"

# Read the existing JSON data from the file
with open(json_file_path, 'r') as json_file:
    existing_data = json.load(json_file)

# New data to add as a new object (a dictionary)
new_object = {"name": "Eve", "age": 22, "city": "Boston"}

# Add the new object to the existing data
existing_data.append(new_object)

# Write the updated data back to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(existing_data, json_file, indent=4)  # Use indent for formatting
