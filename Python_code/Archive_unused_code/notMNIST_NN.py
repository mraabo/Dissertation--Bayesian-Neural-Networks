import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ----------------------------- Loading data ---------------------------------

PATH = 'Python_code/data/notMNIST_large'

classes = os.listdir(PATH)
num_classes = len(classes)

print("There are {} classes: {}".format(num_classes, classes))

# Convert image to matrices and organize matrices for with labels
X = []
y = []

for directory in os.listdir(PATH):
    for image in os.listdir(PATH + '/' + directory):
        try:
            path = PATH + '/' + directory + '/' + image
            img = Image.open(path)
            img.load()
            img_X = np.asarray(img, dtype=np.int16)
            X.append(img_X)
            y.append(directory)
        except:
            None

X = np.asarray(X)
y = np.asarray(y)

num_images = len(X)
size = len(X[0])
print("Shapes of X and y data ", X.shape, y.shape)

# -----------------------------  Visualizing the data  ---------------------------------

# for let in sorted(classes):
#     letter = X[y == let]

#     plt.figure(figsize=(15, 20))
#     for i in range(5):
#         subplot(10, 5, i+1)
#         plt.imshow(letter[i], cmap='gray')
#         plt.title("Letter {}".format(let))

# plt.tight_layout()
# plt.show()

fig, axes = plt.subplots(10, 5, figsize=(15, 20))
for let in sorted(classes):
    letter = X[y == let]
    for i in range(5):
        ax = axes[i//10, i % 5]
        ax.imshow(letter[i], cmap='Blues')
        ax.title("Letter {}".format(let))

plt.tight_layout()
plt.show()


# -----------------------------  Preparing data ---------------------------------------
# Converting labels A-J to number 1-10
y = list(map(lambda x: ord(x) - ord('A'), y))
y = np.asarray(y)

# Shuffling data and splitting it into train and test set
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X = X[indices]
y = y[indices]
num_train_img = np.int(0.7 * X.shape[0])
print("Number of training images:", num_train_img)

X_train = X[:num_train_img]
y_train = y[:num_train_img]

X_test = X[num_train_img:]
y_test = y[num_train_img:]

# Converting y to probabilities
if y_train.ndim == 1:
    y_train = tf.keras.utils.to_categorical(y_train, 10)
if y_test.ndim == 1:
    y_test = tf.keras.utils.to_categorical(y_test, 10)

# Normalize data
if np.max(X_train) == 255:
    X_train = X_train / 255
if np.max(X_test) == 255:
    X_test = X_test / 255

# Reshaping X i.e. flattening the images to 1D
if X_train.ndim == 3:
    num_pixels = size * size
    X_train_1d = X_train.reshape(
        X_train.shape[0], num_pixels).astype('float32')
    X_test_1d = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')

print("Checking dimensions of datasets:\n")
print("X_train shape and X_test shape", X_train.shape, X_test.shape)
input_shape = X_train.shape[1:]
print("input shape", input_shape)
print("X_train_1d shape and X_test_1d shape",
      X_train_1d.shape, X_test_1d.shape)
input_shape_1d = X_train_1d.shape[1]
print("input_shape_1d", input_shape_1d)
print("y_train shape and y_test shape", y_train.shape, y_test.shape)

# -----------------------------  Neural network creation and training ---------------------------------------

# Creating neural network
model = tf.keras.Sequential([
    tf.keras.Input((28, 28), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)
model.evaluate(x_test, y_test)

# -----------------------------  Neural network evalutation ---------------------------------------
score = model.evaluate(X_test_1d, y_test, verbose=1)
print("Accuracy", score)
