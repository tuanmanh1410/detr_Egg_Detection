# Count the number of class with xml files in a directory
# Usage: python Count_Object.py <directory>
# Example: python Count_Object.py /home/user/data

import os
import xml.etree.ElementTree as ET

def count_obj(dir):
    # Create the dictionary for counting object
    obj_dict = {"One": 0, "Two": 0, "Three": 0, "Four": 0, "Five": 0, "Six": 0}
    for file in os.listdir(dir):
        if file.endswith(".xml"):
            #open the file
            tree = ET.parse(dir + "/" + file)
            root = tree.getroot()
            #count the number of objects
            for obj in root.findall('bndbox'):
                #Find the name of the object
                name = obj.find('state').text
                if name == '1':
                    obj_dict["One"] += 1
                elif name == '2':
                    obj_dict["Two"] += 1
                elif name == '3':
                    obj_dict["Three"] += 1
                elif name == '4':
                    obj_dict["Four"] += 1
                elif name == '5':
                    obj_dict["Five"] += 1
                else:
                    obj_dict["Six"] += 1

    return obj_dict

if __name__ == "__main__":
    import sys
    dir = sys.argv[1]
    print(count_obj(dir))
