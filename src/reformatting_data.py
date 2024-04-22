import os
from datetime import datetime

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
            date = filename[:10]
            # Splitting timestamp to handle different formats
            time_parts = timestamp.split(':')
            if len(time_parts) == 2:
                # If format is mm:ss
                timestamp = f"00:{timestamp}"
            elif len(time_parts) == 3:
                # If format is hh:mm:ss
                timestamp = timestamp
            else:
                # Invalid format, skipping this line
                continue
            try:
                timestamp_with_date = datetime.strptime(f"{date} {timestamp}", "%Y-%m-%d %H:%M:%S").strftime("%a %b %d %H:%M:%S %Y %z")
                reformatted_lines.append(f"{name};{timestamp_with_date};{lines[lines.index(line)+1].strip()}")
            except ValueError:
                # Skip the line if timestamp parsing fails
                continue

    return reformatted_lines

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        # Add header to the reformatted lines
        header = "speaker_name;date_time;transcript_text"
        reformatted_lines = process_file(filename)
        reformatted_lines.insert(0, header)

        output_filename = os.path.splitext(filename)[0] + "_rf.csv"
        with open(os.path.join(output_folder, output_filename), 'w') as output_file:
            output_file.write("\n".join(reformatted_lines))
