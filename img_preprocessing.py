import cv2
import numpy as np

# Resize the image to a fixed size
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Normalize the pixel values to a range of 0-1
def normalize_image(image):
    return image / 255.0

# Apply image augmentation techniques
def augment_image(image):
    # Apply random brightness and contrast adjustment
    alpha = np.random.uniform(0.9, 1.1)
    beta = np.random.randint(-10, 10)
    image = cv2.addWeighted(image, alpha, np.zeros(image.shape, dtype=image.dtype), 0, beta)

    # Apply random horizontal flip
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)

    return image

# Preprocess a batch of images
def preprocess_images(images):
    images= cv2.imread(images)
    preprocessed_images = []
    for image in images:
        # Resize the image
        image = resize_image(image)

        # Normalize the pixel values
        image = normalize_image(image)

        # Apply image augmentation techniques
        image = augment_image(image)

        # Add the preprocessed image to the list
        preprocessed_images.append(image)

    # Convert the list of preprocessed images to a NumPy array
    preprocessed_images = np.array(preprocessed_images)

    return preprocessed_images
