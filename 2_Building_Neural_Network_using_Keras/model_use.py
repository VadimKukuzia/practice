import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from random import randrange

# Build the model.
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# Load the model's saved weights.
model.load_weights('models_n_results/model_19.02.2021_18.49.05/model.h5')

test_images = mnist.test_images()
test_labels = mnist.test_labels()

test_images = (test_images / 255) - 0.5
test_images = test_images.reshape((-1, 784))

random = randrange(5, len(test_images))
predictions = model.predict(test_images[random-5:random])


# Print our model's predictions.
print('Predicted values:', np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
# print(test_labels[:5])  # [7, 2, 1, 0, 4]

for i in range(random-5, random):
    image = np.array(test_images[i], dtype=float)
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
