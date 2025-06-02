#---------------------------------------------------------
# Import dependencies
#---------------------------------------------------------
import os, shutil, subprocess


# run command
def run_command(command):
    """
    Execute a command in the shell.
    """
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


# get current working directory (to make sure the code works anywhere in your device)
cwd = os.getcwd()

# get parent directory (because we are in scripts direcotry)
parent_dir = os.path.dirname(cwd)


#---------------------------------------------------------
# Restructure the dataset for segmentation and diffusion
#---------------------------------------------------------
print('\n\n#-----------------------------------------------------------------------------------')
print('Restructuring the dataset into directory "detect" for detection/diffusion')
print('#-----------------------------------------------------------------------------------\n')

# remove directory if exist
if os.path.exists('detect'):
    shutil.rmtree('detect') 

# create directory
os.makedirs(f'{parent_dir}/detect/train/image')
os.makedirs(f'{parent_dir}/detect/train/label')

os.makedirs(f'{parent_dir}/detect/val/image')
os.makedirs(f'{parent_dir}/detect/val/label')

# convert from coco-seg format to yolo-seg format
command = f'python3 coco2yolo/coco2yolo -ann-path "{parent_dir}/data/train/_annotations.coco.json" -img-dir "{parent_dir}/data/train" -task-dir "{parent_dir}/detect/train/image" -set union'
run_command(command)

command = f'python3 coco2yolo/coco2yolo -ann-path "{parent_dir}/data/valid/_annotations.coco.json" -img-dir "{parent_dir}/data/valid" -task-dir "{parent_dir}/detect/val/image" -set union'
run_command(command)


# Source and destination directories
src_dir = f"{parent_dir}/detect/train/image"
dst_dir = f"{parent_dir}/detect/train/label"

# Iterate through files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.txt'):
        # Construct full file paths
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # Copy the file
        shutil.copy(src_path, dst_path)

        # Delete the file from the source directory
        os.remove(src_path)


# Source and destination directories
src_dir = f"{parent_dir}/detect/val/image"
dst_dir = f"{parent_dir}/detect/val/label"

# Iterate through files in the source directory
for filename in os.listdir(src_dir):
    if filename.endswith('.txt'):
        # Construct full file paths
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        # Copy the file
        shutil.copy(src_path, dst_path)

        # Delete the file from the source directory
        os.remove(src_path)

#---------------------------------------------------------
# Restructuring the dataset for YOLOv8 Training
#---------------------------------------------------------
print('\n\n#-----------------------------------------------------------------------------------')
print('Restructuring the dataset into directory "mydata" for YOLOv8 training')
print('#-----------------------------------------------------------------------------------\n')

# creating the directory
print('Creating the "mydata" directory ...\n')
os.makedirs(f'{parent_dir}/mydata')
os.makedirs(f'{parent_dir}/mydata/images')
os.makedirs(f'{parent_dir}/mydata/labels')

os.makedirs(f'{parent_dir}/mydata/images/train')
os.makedirs(f'{parent_dir}/mydata/images/val')
os.makedirs(f'{parent_dir}/mydata/labels/train')
os.makedirs(f'{parent_dir}/mydata/labels/val')


# copy the content from "detect" directory
print('Copying the contents from "detect" ...\n')
shutil.copytree(f'{parent_dir}/detect/train/image', f'{parent_dir}/mydata/images/train', dirs_exist_ok=True)
shutil.copytree(f'{parent_dir}/detect/val/image', f'{parent_dir}/mydata/images/val', dirs_exist_ok=True)

shutil.copytree(f'{parent_dir}/detect/train/label', f'{parent_dir}/mydata/labels/train', dirs_exist_ok=True)
shutil.copytree(f'{parent_dir}/detect/val/label', f'{parent_dir}/mydata/labels/val', dirs_exist_ok=True)


# get ready the yaml file for the dataset
print('Creating the YOLO yaml file for the dataset ...\n')
yaml_config = [
    f"path: {parent_dir}/mydata",
    "train: images/train",
    "val: images/val",
    "",
    "names: ",
    "  0: 'small-object'",
    "  1: 'basketball field'",
    "  2: 'building'",
    "  3: 'crosswalk'",
    "  4: 'football field'",
    "  5: 'graveyard'",
    "  6: 'large vehicle'",
    "  7: 'medium vehicle'",
    "  8: 'playground'",
    "  9: 'roundabout'",
    "  10: 'ship'",
    "  11: 'small vehicle'",
    "  12: 'swimming pool'",
    "  13: 'tennis court'",
    "  14: 'train'",

]

# save annotation as txt file
yaml_file = f'{parent_dir}/mydata/data.yaml'
with open(yaml_file, 'w') as file:
    for item in yaml_config:
        file.write(item + '\n')

print('Done setting up the dataset!')
