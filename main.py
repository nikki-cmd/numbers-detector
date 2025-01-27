import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from model import LogRegModel
from keras.models import load_model

# Загрузка сохранённого датасета
data = np.load("mnist_numbers.npz")
images = data['images']
labels = data['labels']

# Нормализация изображений (если нужно)
images = images / 255.0  # Приведение значений пикселей в диапазон [0, 1]

# Разделение на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Размер тренировочной выборки:", x_train.shape, y_train.shape)
print("Размер тестовой выборки:", x_test.shape, y_test.shape)

model = load_model("models/model.keras")

for i in range(0, len(images[0])):
    print(images[0][i])

for i in range(5):
    
    print(images[i].shape)
    plt.imshow(images[i], cmap='gray')
    plt.show()


'''x_train_vectors = [[] for _ in range(len(x_train))]
x_test_vectors = [[] for _ in range(len(x_test))]

for pic in range(0, len(x_train)):
    for i in range(0, len(x_train[pic])):
        for j in range(0, len(x_train[pic][i])):
            x_train_vectors[pic].append(x_train[pic][i][j])
        
for pic in range(0, len(x_test)):
    for i in range(0, len(x_test[pic])):
        for j in range(0, len(x_test[pic][i])):
            x_test_vectors[pic].append(x_test[pic][i][j])'''



'''for i in range(5):
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Number: {y_train[i]}")
    plt.show()'''


