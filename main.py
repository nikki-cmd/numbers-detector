import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import LogRegModel
from keras.models import load_model

def chunks(lst, n):
    squares = []
    for i in range(0, len(lst), n):
        square = lst[i:i + n]
        squares.append(square)
    return squares

data = np.load("mnist_numbers.npz")
images = data['images']
labels = data['labels']

images = images / 255.0 

model = load_model("models/model.keras")


input_len = images[0].shape[1] // 28


for picture in range(10):
    image = images[picture]

    single_images = []

    for i in range(input_len):
        single = [row[i*28:(i+1)*28] for row in image]
        single_images.append(single)
    print(len(single_images))

    single_images_expanded = [np.expand_dims(img, axis=-1) for img in single_images]

    single_images_array = np.array(single_images_expanded)

    res = ""

    for i in range(single_images_array.shape[0]):
        prediction = model.predict(np.expand_dims(single_images_array[i], axis=0))
        res += str(np.argmax(prediction))
        
    plt.imshow(image, cmap='gray')  
    plt.title(res)
    plt.show()