import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Model, Sequential

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from keras.layers import SimpleRNN
from keras.layers import Dense, Input

warnings.filterwarnings('ignore')

mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)



list(filter(lambda x:x != 0, train_images[0].reshape(-1)))[:10]

print(test_images[test_images != 0][:10])
print(test_images[0].shape)
print(test_images.dtype)

print(train_images.max())
print(train_images.min())
print(train_labels.max())
print(train_labels.min())

print(test_images.max())
print(test_images.min())
print(test_labels.max())
print(test_labels.min())


train_images = train_images.astype(np.float64)
test_images = test_images.astype(np.float64)

print(train_images.dtype)
print(test_images.dtype)


train_images = train_images / train_images.max()
test_images = test_images / test_images.max()

print(test_images[test_images != 0][:10])
print(test_images[0].shape)
print(test_images.dtype)

print(train_images.max())
print(train_images.min())
print(train_labels.max())
print(train_labels.min())

print(test_images.max())
print(test_images.min())
print(test_labels.max())
print(test_labels.min())


np.hstack(train_images[:5]).shape
plt.imshow(train_images[:5].transpose(1, 0, 2).reshape(28, -1), cmap='gray')
plt.show()


train_noisy_image = train_images + np.random.normal(0.5, 0.1, (28, 28))
train_noisy_image[train_noisy_image > 1] = 1

test_noisy_image = test_images + np.random.normal(0.5, 0.1, (28, 28))
test_noisy_image[test_noisy_image > 1] = 1

plt.imshow(train_noisy_image[0])
plt.colorbar()
plt.show()


from keras.utils import to_categorical

train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)


print(train_labels.shape)
print(train_labels[:5])
plt.imshow(train_images[:5].transpose(1, 0, 2).reshape(28, -1))
plt.show()

print(test_labels.shape)
print(test_labels[0])




  
input = Input(shape=(28, 28))
x1 = SimpleRNN(units = 64, activation = 'tanh')(input)
x2 = Dense(units = 10, activation = 'softmax')(x1)

model = Model(input, x2)
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


hist = model.fit(
	train_noisy_image,
	train_labels,
	
	validation_data = (
		test_noisy_image, 
		test_labels
	),
	epochs = 5,
	verbose = 1
)

plt.plot(hist.history['loss'], label = 'loss')
plt.plot(hist.history['accuracy'], label = 'accuracy')
plt.plot(hist.history['val_loss'], label = 'val_loss')
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
plt.legend(loc = 'upper left')
plt.show()

res = model.predict(test_noisy_image[:1])
print(res.shape)
print(res)


imgs = np.concatenate([test_noisy_image[0], test_images[0]], axis = 1)

plt.imshow(imgs)
plt.show()

res[0].argmax()

plt.bar(range(10), res[0], color='red')
plt.bar(np.array(range(10)) + 0.1, test_labels[0])
plt.show()

