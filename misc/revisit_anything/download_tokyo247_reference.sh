#!/bin/bash

# Base URL for the files
base_url="https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Tokyo247/database_gsv_vga"

# Loop to download .tar files from 03814 to 03829
for i in $(seq -w 03814 03829); do
    echo "Downloading ${i}.tar..."
    if ! wget "${base_url}/${i}.tar"; then
        echo "Failed to download ${i}.tar"
        continue  # Skip this iteration and continue with the next
    fi
done


# Download miscellaneous files
wget "${base_url}/Readme.txt"
wget "${base_url}/dbfnames.mat"
wget "${base_url}/md5sum_list.txt"


# print statement to check if ALL the files are download
echo "CHECK manually if all files downloaded from 03814 to 03829"

# Loop to untar all .tar files and move the extracted contents
for tar_file in *.tar; do
    echo "Untarring $tar_file..."
    # Extract the tar file
    tar -xf "$tar_file"
done


# CODE for "tokyo247.mat" 

# URL of the tar.gz file
url="https://www.di.ens.fr/willow/research/netvlad/data/netvlad_v100_datasets.tar.gz"

# Target directory for extraction
extract_dir="./misc_mat_files"

# Filename to copy
mat_file="tokyo247.mat"

# Ensure the extraction directory exists
mkdir -p "$extract_dir"

# Download the file
echo "Downloading the file..."
wget "$url" -O "netvlad_v100_datasets.tar.gz"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download successful, extracting..."
    # Extract the tar.gz file to the specified directory
    tar -xzf "netvlad_v100_datasets.tar.gz" -C "$extract_dir"
    
    # Check if the extraction was successful
    if [ $? -eq 0 ]; then
        echo "Extraction successful. Copying the ${mat_file} file..."
        # Copy the .mat file to the current directory
		cp "${extract_dir}/datasets/${mat_file}" .
        
        # Verify if the copy was successful
        if [ $? -eq 0 ]; then
            echo "${mat_file} has been copied successfully."
        else
            echo "Failed to copy ${mat_file}."
        fi
    else
        echo "Failed to extract files."
    fi
else
    echo "Failed to download the file."
fi



# Move from here to vpr_datasets_downloader repo

## Target directory where extracted folders will be moved
#target_directory="/scratch/saishubodh/segments_data/VPR-datasets-downloader/datasets/tokyo247/raw_data/tokyo247/"
#
## Create the target directory if it doesn't exist
#mkdir -p "$target_directory"
#
## Loop to untar all .tar files and move the extracted contents
#for tar_file in *.tar; do
#    # Get the directory name from the tar file
#    directory_name=$(basename "$tar_file" .tar)
#    echo "Moving $directory_name to $target_directory"
#    # Move the extracted directory to the target directory
#    mv "$directory_name" "$target_directory"
#done