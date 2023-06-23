import img_preprocessing as imgp
import xml.etree.ElementTree as ET
import numpy as np
import os
from PIL import Image






# Load annotations
def load_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        annotations.append([x1, y1, x2, y2])
    return np.array(annotations)



# Generate training data
def generate_training_data(image_file, xml_file):
    
    image = imgp.preprocess_images(image_file)
    annotations = load_annotations(xml_file)
    # Create bounding boxes around the number plates
    labels = np.zeros((annotations.shape[0], 5))
    labels[:, 1:] = annotations
    # Normalize the coordinates
    labels[:, 1] /= image.shape[0]
    labels[:, 2] /= image.shape[1]
    labels[:, 3] /= image.shape[0]
    labels[:, 4] /= image.shape[1]
    return image, labels

