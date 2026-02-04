import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
import re

class Core:
    def save_model(self, file_name="my_model.keras"):
        self.model.save(file_name)
        print(f"Модель сохранена как {file_name}")

    def load_model(self, file_name="my_model.keras"):
        self.model = models.load_model(file_name)
        print("Модель успешно загружена")


class CNN:
    def __init__(self,classes = 10,batch_size = 8,filters_count = 32,kernel_size = (3,3),image_w = 32, image_h = 32, color_channels = 3):

        self.classes = classes
        self.filters_count = filters_count
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.image_w = image_w
        self.image_h = image_h
        self.picture_shape = (image_h,image_w,color_channels)

    def build_model(self):
        self.model = models.Sequential([
            layers.Input(shape=self.picture_shape),

            layers.Conv2D(filters=self.filters_count, kernel_size=self.kernel_size, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(filters=self.filters_count * 2, kernel_size=self.kernel_size, activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),

            layers.Dense(self.filters_count * 2, activation='relu'),
            layers.Dense(self.classes, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self.model


    def create_dataset(self,path):
        path = path.replace('\\','/')

        train_ds = tf.keras.utils.image_dataset_from_directory(
            path + '/',
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.image_h, self.image_w),
            batch_size=self.batch_size
        )

        return train_ds

    def fit_model(self,train_ds,epochs):
        self.model.fit(
            train_ds,
            epochs = epochs,
            validation_data=train_ds,
        )

    def predict(self,path,model,classes):
        img_path = path
        img = image.load_img(img_path, target_size=(self.image_h, self.image_w))

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        print(predictions)
        score = tf.nn.softmax(predictions[0])
        class_names = classes

        return class_names[np.argmax(score)], (100 * np.max(score))

# path = 'C:\\Users\\Yura\\PycharmProjects\\PythonProjectSchool\\TestTrainingDataset\\data'
#
# classes = ['with_mask', 'without_mask']
# cnn = CNN(classes=2, image_w=32, image_h=32,batch_size=64)
#
# model = cnn.build_model()
#
# train_data = cnn.create_dataset(path)
#
# cnn.model.fit(
#     train_data,
#     validation_data=train_data,
#     epochs=3
# )
#
# test_img = 'D:\PyProjects\Model_it\Test\\test.png'
# test_img2 = 'D:\PyProjects\Model_it\Test\channels4_profile.jpg'
# print(cnn.predict(test_img, model, classes))
# print(cnn.predict(test_img2, model, classes))
# model.save('mask_model3.keras')


class LINEAR:
    def __init__(self, input_shape=2, hidden_shape=16, output_shape=1, batch_size=2):
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape
        self.output_shape = output_shape
        self.batch_size = batch_size
        self.model = None

    def build_model(self):
        self.model = models.Sequential([
            layers.Input(shape=(self.input_shape,)),
            layers.Dense(self.hidden_shape, activation='relu'),
            layers.Dense(self.hidden_shape, activation='relu'),
            layers.Dense(self.output_shape)
        ])
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model

    def create_dataset(self, path, expected_shape):
        path = path.replace('\\', '/')
        data = []
        with open(path, 'r') as f:
            # Читаем весь файл и находим ВСЕ числа, игнорируя текст
            content = f.read()
            # Убираем перед поиском чисел
            content = re.sub(r'\\', '', content)
            numbers = re.findall(r"[-+]?\d*\.\d+|\b[-+]?\d+\b", content)

            float_numbers = [float(x) for x in numbers]

            # Нарезаем плоский список чисел на нужные группы
            for i in range(0, len(float_numbers), expected_shape):
                chunk = float_numbers[i:i + expected_shape]
            if len(chunk) == expected_shape:
                data.append(chunk)

        return np.array(data)

    def fit_model(self, x_train, y_train, epochs):
        # Важно: X и Y должны быть одинаковой длины
        min_len = min(len(x_train), len(y_train))
        self.model.fit(
            x_train[:min_len],
            y_train[:min_len],
            epochs=epochs,
            batch_size=self.batch_size,
            verbose=1
        )


# l = LINEAR(input_shape=2, hidden_shape=16, output_shape=1, batch_size=8)
# model = l.build_model()
#
# X = l.create_dataset('testtxt.txt', expected_shape=2)
# Y = l.create_dataset('testRes.txt', expected_shape=1)
#
# print(f"Загружено X: {len(X)} примеров, Y: {len(Y)} ответов")
#
# l.fit_model(X, Y, epochs=300)
#
# test_val = np.array([[1.0, 1.5]])
# prediction = model.predict(test_val, verbose=0)
# print(f"\nТест [1, 2] -> Предсказание: {prediction[0][0]:.4f} (Ожидалось: 3.0)")
#
# model.save('test_function.keras')

