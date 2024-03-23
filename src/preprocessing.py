import os
import re

# Loading data
def load_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(directory + filename, 'r') as file:
                data.append(file.read())
    return data

# remove characters matching pattern of [.*]
def remove_brackets(data):
    return [re.sub('\n\[.*?\]. ', '', text) for text in data]

# get list of file names and remove the .txt extension
def get_filenames(directory):
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filenames.append(filename[:-4])
    return filenames

# save data as individual txt files.
def save_data(data, directory, filenames):
    for text, name in zip(data, filenames):
        with open(directory + name + '_cleaned' + '.txt', 'w') as file:
            file.write(text)

data = load_data("/data/raw_transcripts/")

data = remove_brackets(data)

filenames = get_filenames("/data/raw_transcripts/")

save_data(data, "/data/cleaned_transcripts/", filenames)