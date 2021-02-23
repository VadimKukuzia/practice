import keras
import numpy as np
import mnist
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from random import randrange

from tensorflow.keras.layers import Flatten
model = keras.models.load_model('models_n_results/model_23.02.2021_22.26.08/')
# model = Sequential([
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax'),
# ])

# Load the model's saved weights.
# model.load('models_n_results/model_23.02.2021_21.31.28/model.h5')

test_images = mnist.test_images()
test_labels = mnist.test_labels()


test_images = tf.keras.utils.normalize(test_images, axis=1)  # scales data between 0 and 1

random = randrange(5, len(test_images))
predictions = model.predict(test_images[random-5:random])

print('Valid values:', test_labels[random-5:random])
# Print our model's predictions.
print('Predicted values:', np.argmax(predictions, axis=1))



for i in range(random-5, random):
    image = np.array(test_images[i], dtype=float)
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
