
from PIL import Image, ImageDraw as D
import xml.etree.ElementTree as ET
import glob
import os


# Preprocess image - Get list of images
def collect_all_images(dir_test):
    """
    Function to return a list of image paths.

    :param dir_test: Directory containing images or single image path.

    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images

def Get_GroundTruth_Bounding_Boxes(file_name):
    # Get information from the xml file
    tree = ET.parse(file_name)
    root = tree.getroot()
    # Read information from <object_count> tag
    object_count = int(root.find('object_count').text)
    Bbox_GT = []

    for member in root.findall('bndbox'):
        xmin = float(member.find('x_min').text)
        ymin = float(member.find('y_min').text)
        xmax = float(member.find('x_max').text)
        ymax = float(member.find('y_max').text)
        bndbox = [xmin, ymin, xmax, ymax]
        # Append bounding box to the Bbox_GT list
        Bbox_GT.append(bndbox)
    
    return object_count, Bbox_GT

# Main function
def main(source_dir, target_dir):
    # Check target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Get list of images
    test_images = collect_all_images(source_dir)

    count = 0
    # Loop through all images
    for image in test_images:
        if count == 20:
            break
        # Replace the path of the image with the path of the xml file
        xml_file = image.replace('jpg', 'xml')
        # Get the number of objects and bounding boxes from the xml file
        object_count, Bbox_GT = Get_GroundTruth_Bounding_Boxes(xml_file)
        # Open the image
        im = Image.open(image)
        # Draw the bounding boxes
        draw = D.Draw(im)
        for j in range(object_count):
            draw.rectangle(Bbox_GT[j], outline="red", width=4)
        # Save the image on the target directory
        im.save(target_dir + "/" + os.path.basename(image))
        count += 1

if __name__ == "__main__":
    # Get the source directory
    source_dir = "/hdd/ttmanh/detr_Egg_Detection/Bbox_Images"
    # Get the target directory
    target_dir = "/hdd/ttmanh/detr_Egg_Detection/Bbox_Images/Demo"
    # Call the main function
    main(source_dir, target_dir)