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

# Разделение на тренировочную и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print("Размер тренировочной выборки:", x_train.shape, y_train.shape)
print("Размер тестовой выборки:", x_test.shape, y_test.shape)

model = load_model("models/model.keras")

images = images[0]

# Список для хранения отдельных изображений
single_images = []

# Разделяем images[0] на 4 изображения 28x28
for i in range(4):
    single = [row[i*28:(i+1)*28] for row in images]
    single_images.append(single)

# Преобразуем каждое изображение в форму (28, 28, 1), добавив ось каналов
single_images_expanded = [np.expand_dims(img, axis=-1) for img in single_images]

# Объединяем все изображения в один массив с формой (4, 28, 28, 1)
single_images_array = np.array(single_images_expanded)

# Проверяем форму
print(single_images_array.shape)  # Должно быть (4, 28, 28, 1)

d={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

# Делаем предсказания для каждого изображения
for i in range(single_images_array.shape[0]):
    prediction = model.predict(np.expand_dims(single_images_array[i], axis=0))
    print(f"Predictions for image {i+1}:", np.argmax(prediction))
    
    # Отображаем изображение
    plt.imshow(single_images_array[i, :, :, 0], cmap='gray')  # Убираем ось каналов для отображения
    plt.title(f'Image {i+1}')
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


