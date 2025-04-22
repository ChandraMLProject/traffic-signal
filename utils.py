
import numpy as np
from PIL import Image
import pandas as pd

def preprocess_image(image):
    image = image.resize((32, 32))
    return np.array(image) / 255.0

def get_class_name(class_id):
    labels = pd.read_csv("labels.csv")
    return labels.loc[class_id, "Name"]
