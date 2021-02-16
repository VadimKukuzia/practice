import numpy as np
import mnist
from matplotlib import ticker
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# Build the model.
model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
print(model.summary())

# Save model summary



# Compile the model.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)
# TODO: Побаловаться с Tuning Hyperparameters

# Train the model.
history = model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
    validation_split=0.1,
)


print('Train accuracy: Min - ', min(history.history['accuracy']), ' |', 'Max - ', max(history.history['accuracy']))
print('Validation accuracy: Min - ', min(history.history['val_accuracy']), ' |', 'Max - ', max(history.history['val_accuracy']))
print('Train loss: Max - ', max(history.history['loss']), ' |', 'Min - ', min(history.history['loss']))
print('Validation loss: Max -', max(history.history['val_loss']), ' |', 'Min - ', min(history.history['val_loss']))
# print(history.history.keys())
# print(history.history['accuracy'])
# print(history.history['val_accuracy'])
# print(history.history['loss'])
# print(history.history['val_loss'])


fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Model accuracy and loss')
fig.set_figheight(5)
fig.set_figwidth(10)

ax1.grid()
ax2.grid()

# ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# #  Устанавливаем интервал вспомогательных делений:
# ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#
# ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
# #  Устанавливаем интервал вспомогательных делений:
# ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])

ax1.set_title(f'Model accuracy. Train min = ')
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(['Train', 'Validation'], loc='best')

ax2.plot(history.history['loss'])
ax2.plot(history.history['val_loss'])

ax2.set_title('Model loss')
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(['Train', 'Validation'], loc='best')

plt.show()
#  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()

# Evaluate the model.
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

with open('models_n_results/model.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

# Save the model to disk.
model.save_weights('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1))  # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5])  # [7, 2, 1, 0, 4]
