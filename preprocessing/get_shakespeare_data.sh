#!/bin/bash

# Check if the directory argument is provided
if [ $# -ne 1 ]; then
  echo "Usage: $0 <directory_name>"
  exit 1
fi

# Store the directory name provided as an argument
directory_name="$1"

# Check if the directory exists, and if not, create it
if [ ! -d "$directory_name" ]; then
  mkdir -p "$directory_name"
fi

# Navigate to the specified directory
cd "$directory_name" || exit 1

# URL of the Google Drive file you want to download
train_file_id='1iUMuFzn9zMAn0WDJtzyGciJEmyu6V80t'
test_file_id='1YSG2PPlqMZb3PW4LA6HgMWnG7XhvWKg4'

# Output file name
train_file='shakespare_train_processed.pkl'
test_file='shakespare_test_processed.pkl'

# Download the file using gdown
gdown "$train_file_id" -O "$train_file"
gdown "$test_file_id" -O "$test_file"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download completed successfully."
else
  echo "Download failed."
fi
