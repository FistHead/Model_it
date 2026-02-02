import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image
from modelMaker import *

cnn = CNN(image_w = 64, image_h = 64)

model = cnn.build_model()
model = models.load_model("mask_model2.keras")

img= 'test.png'
classes = ['with_mask', 'without_mask']
print(cnn.predict(img, model, classes))


lnn = LINEAR()
model2 = lnn.build_model()
model2 = models.load_model("test_function.keras")
test_input = np.array([[2,4]])
print(model2.predict(test_input).round())