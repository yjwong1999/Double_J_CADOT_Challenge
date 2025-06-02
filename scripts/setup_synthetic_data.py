#---------------------------------------------------------
# Import dependencies
#---------------------------------------------------------
import os, shutil
import shutil

# get current working directory (to make sure the code works anywhere in your device)
cwd = os.getcwd()

# get parent directory (because we are in scripts direcotry)
cwd = os.path.dirname(cwd)


#---------------------------------------------------
# copy files from "synthetic_data" directory to "mydata" directory
#---------------------------------------------------

# Define source and destination directories
source_folder = f"{cwd}/synthetic_data"
destination_folder = f"{cwd}/mydata/images/train"

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# Copy all files
for filename in os.listdir(source_folder):
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(destination_folder, filename)

    if os.path.isfile(src_path):  # Ensure it's a file
        shutil.copy2(src_path, dst_path)

print(f"All files copied from {source_folder} to {destination_folder}")


#---------------------------------------------------
# copy label
#---------------------------------------------------
import os
import shutil

# Define source folder
source_folder = f"{cwd}/mydata/labels/train"

# Iterate through all files in the folder
for filename in os.listdir(source_folder):
    src_path = os.path.join(source_folder, filename)

    if os.path.isfile(src_path):  # Ensure it's a file
        new_filename = f"condon_{filename}"  # Prepend 'condon_' to the filename
        dst_path = os.path.join(source_folder, new_filename)

        shutil.copy2(src_path, dst_path)  # Copy file while preserving metadata

print("Copies created successfully!")
