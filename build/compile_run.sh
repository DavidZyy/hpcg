#!/bin/bash

# Function to check and create directory if it doesn't exist
check_and_create_dir() {
    local dir_name=$1
    if [ ! -d "$dir_name" ]; then
        echo "Directory $dir_name does not exist. Creating..."
        mkdir "$dir_name"
        if [ $? -eq 0 ]; then
            echo "Directory $dir_name created successfully."
        else
            echo "Failed to create directory $dir_name." >&2
            exit 1
        fi
    fi
}

# Check and create src directory
check_and_create_dir "src"

# Check and create bin directory
check_and_create_dir "bin"


make
./run.sh
