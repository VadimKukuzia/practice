from datetime import datetime
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import os

from tensorflow.keras.optimizers import Adamax

from tensorflow.keras.layers import Flatten


mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

train_images = tf.keras.utils.normalize(train_images, axis=1)  # scales data between 0 and 1
test_images = tf.keras.utils.normalize(test_images, axis=1)  # scales data between 0 and 1


# Build the model.
model = Sequential([
    Flatten(),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax'),
])

# Save model summary


# Compile the model.
model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
print(train_images.shape, train_labels.shape)

# Train the model.
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)

# time for result saving. creating new model directory
time = datetime.strftime(datetime.now(), "%d.%m.%Y_%H.%M.%S")
os.mkdir(f'models_n_results/model_{time}')

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model accuracy and loss')
fig.set_figheight(5)
fig.set_figwidth(10)

ax1.grid()
ax2.grid()


ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])

ax1.set_title(f'Model accuracy')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='best')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])

ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='best')

plt.savefig(f'models_n_results/model_{time}/Accuracy_loss.png')
plt.show()


# Evaluate the model.
print('Model evaluate:')
model.evaluate(
    test_images,
    test_labels
)

with open(f'models_n_results/model_{time}/model_evaluate.txt', 'w') as f:
    with redirect_stdout(f):
        model.evaluate(
            test_images,
            test_labels
        )


with open(f'models_n_results/model_{time}/model.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Save the model to disk.
model.save_weights(f'models_n_results/model_{time}/model.h5')
