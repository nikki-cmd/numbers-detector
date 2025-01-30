import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import LogRegModel
from keras.models import load_model
import pickle

def chunks(lst, n):
    squares = []
    for i in range(0, len(lst), n):
        square = lst[i:i + n]
        squares.append(square)
    return squares

with open("mnist_numbers.pkl", "rb") as f:
    data = pickle.load(f)
    images = data["images"]
    labels = data["labels"]

model = load_model("models/model.keras")

for picture in range(10):

    image = images[picture]
    input_len = image.shape[1] // 28
    single_images = []

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