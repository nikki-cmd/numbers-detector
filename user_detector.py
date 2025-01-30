import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import LogRegModel
from keras.models import load_model
from PIL import Image, ImageOps


image = Image.open('pictures/photo_2025-01-30_17-17-29.jpg')
image = image.convert('L')

width, height = image.size
image = ImageOps.invert(image)

image = np.array(image)

image = image / 255.0

input_len = image.shape[1] // 28
single_images = []


model = load_model("models/model.keras")

for i in range(input_len):
    single = [row[i*28:(i+1)*28] for row in image]
    single_images.append(single)

single_images_expanded = [np.expand_dims(img, axis=-1) for img in single_images]

single_images_array = np.array(single_images_expanded)

res = ""

for i in range(single_images_array.shape[0]):
    prediction = model.predict(np.expand_dims(single_images_array[i], axis=0))
    res += str(np.argmax(prediction))
        
plt.imshow(image, cmap='gray')  
plt.title(f"Predicted: {res}")
plt.show()