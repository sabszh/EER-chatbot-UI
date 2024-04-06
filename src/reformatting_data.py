import os

input_folder = "data/raw_transcripts"
output_folder = "data/reformatted_transcripts"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to process each file
def process_file(filename):
    with open(os.path.join(input_folder, filename), 'r') as file:
        lines = file.readlines()

    reformatted_lines = []
    for line in lines:
        parts = line.strip().split('  ')
        if len(parts) == 2:
            name, timestamp = parts
            date = filename[:10]  # Extracting the date from the filename
            timestamp_with_date = f"{date}, {timestamp}"
            reformatted_lines.append(f"{name}, {timestamp_with_date}, {lines[lines.index(line)+1].strip()}")

    return reformatted_lines

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        reformatted_lines = process_file(filename)
        output_filename = os.path.splitext(filename)[0] + "_rf.txt"
        with open(os.path.join(output_folder, output_filename), 'w') as output_file:
            output_file.write("\n".join(reformatted_lines))
