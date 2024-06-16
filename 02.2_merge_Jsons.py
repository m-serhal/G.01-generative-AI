import json
import os

# Directory containing JSON files to merge
directory = "knowledge_pool"

# Initialize an empty list to store the merged content
merged_data = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        file_path = os.path.join(directory, filename)
        try:
            # Open each JSON file and load its contents
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Check if data is a list containing a dictionary, and if so, extend merged_data with that dictionary
                if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                    merged_data.extend(data)
                # If data is a single dictionary, append it directly
                elif isinstance(data, dict):
                    merged_data.append(data)
                else:
                    print(f"Ignoring unexpected data format in {file_path}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
        except FileNotFoundError as e:
            print(f"File not found: {file_path}: {e}")

# Output file path for merged JSON
output_file = os.path.join(directory, "merged_output.json")

# Write the merged data to the output file
try:
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
    print(f"Successfully merged JSON files into {output_file}")
except Exception as e:
    print(f"Error writing merged data to {output_file}: {e}")
