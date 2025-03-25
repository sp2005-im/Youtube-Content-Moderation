# YouTube Metadata CSV

This repository contains a folder named `new_folder` that holds a CSV file containing YouTube video metadata. The file includes important information about 600 YouTube videos, including video ID, title, description, channel title, publish date, view count, like count, and category.

## Files in this Folder

- `metadata.csv`: This CSV file contains metadata for 600 YouTube videos. The data is structured in 8 columns and includes key information to analyze video performance, categorization, and more.

## How to Use the CSV File

1. **Download the CSV File**:
   - To download the file, click on `metadata.csv` in the folder and select `Download`.

2. **Load the Data in Python**:
   You can load the CSV file into a pandas DataFrame in Python for further analysis or processing. Here's an example code:

   ```python
   import pandas as pd
   
   # Load the YouTube metadata CSV file
   data = pd.read_csv(FILE_PATH) # Replace FILE_PATH with the path of the file (depending on where you are storing it)
   
   # Display the first few rows of the data
   print(data.head())
