# Copy image and xml files with same name from one directory to another
# Select random with specified number of files

import os
import shutil
import random

def copy_files(source_dir, target_dir, num_files):
    # Find image files in source directory
    files = [file for file in os.listdir(source_dir) if file.endswith('.jpg')]
    # Select random files
    random_files = random.sample(files, num_files)
    # Copy files to target directory
    for file in random_files:
        shutil.copy(source_dir + file, target_dir + file)
        # Copy corresponding xml file
        shutil.copy(source_dir + file[:-4] + '.xml', target_dir + file[:-4] + '.xml')
    
# Main function
if __name__ == '__main__':
    # Source directory
    source_dir = '/SSD1/ttmanh/COLOR_COCO/test/'
    # Target directory
    target_dir = '/SSD1/ttmanh/kkokkobot_Detection/detr_Egg_Detection/Final_Demo_COLOR/'
    # Number of files to copy
    num_files = 50
    # Copy files
    copy_files(source_dir, target_dir, num_files)