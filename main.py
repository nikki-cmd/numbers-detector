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

# Загрузка сохранённого датасета
data = np.load("mnist_numbers.npz")
images = data['images']
labels = data['labels']

# Нормализация изображений (если нужно)
images = images / 255.0  # Приведение значений пикселей в диапазон [0, 1]

model = load_model("models/model.keras")


input_len = images[0].shape[1] // 28


for picture in range(10):
    image = images[picture]

    # Список для хранения отдельных изображений
    single_images = []

    # Разделяем images[0] на 4 изображения 28x28
    for i in range(input_len):
        single = [row[i*28:(i+1)*28] for row in image]
        single_images.append(single)
    print(len(single_images))

    # Преобразуем каждое изображение в форму (28, 28, 1), добавив ось каналов
    single_images_expanded = [np.expand_dims(img, axis=-1) for img in single_images]

    # Объединяем все изображения в один массив с формой (4, 28, 28, 1)
    single_images_array = np.array(single_images_expanded)

    res = ""

    for i in range(single_images_array.shape[0]):
        prediction = model.predict(np.expand_dims(single_images_array[i], axis=0))
        res += str(np.argmax(prediction))
        
    plt.imshow(image, cmap='gray')  
    plt.title(res)
    plt.show()
    


'''for i in range(5):
    
    print(images[i].shape)
    plt.imshow(images[i], cmap='gray')
    plt.show()'''


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


