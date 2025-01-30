import numpy as np
import random
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import pickle

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def create_number_from_digits(x_data, y_data, samples=1000):
    images = []
    labels = []
    
    for _ in range(samples):
        num_digits = random.randint(2, 9) 
        indices = np.random.choice(len(x_data), num_digits, replace=False)
        selected_images = x_data[indices]
        selected_labels = y_data[indices]
        
        number = int("".join(map(str, selected_labels)))
        
        combined_image = np.hstack(selected_images)
        
        images.append(combined_image)
        labels.append(number)
    
    return images, labels

samples = 1000  
new_images, new_labels = create_number_from_digits(x_train, y_train, samples)

for i in range(5):
    plt.imshow(new_images[i], cmap='gray')
    plt.show()

with open("mnist_numbers.pkl", "wb") as f:
    pickle.dump({"images": new_images, "labels": new_labels}, f)

