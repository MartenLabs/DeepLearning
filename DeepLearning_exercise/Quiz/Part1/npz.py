from tensorflow import keras
import numpy as np

# datasets = keras.datasets.mnist

# (train_images, train_labels), (test_images, test_labels) = datasets.load_data()

# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# # np.savez('./mnist.npz', train_images = train_images, train_labels = train_labels, 
# #         test_images = test_images, test_labels = test_labels)

# datasets = np.load('./mnist.npz')
# print(list(datasets.keys()))

# train_images = datasets['train_images']
# train_labels = datasets['train_labels']

# test_images = datasets['test_images']
# test_labels = datasets['test_labels']

# print(train_images.shape)
# print(train_labels.shape)

# print(test_images.shape)
# print(test_labels.shape)



fashion_mnist = keras.datasets.fashion_mnist
data = fashion_mnist.load_data()

(train_images, train_labels), (test_images, test_labels) = data

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

np.savez('./fashion_mnist.npz', train_images = train_images, train_labels = train_labels, 
        test_images = test_images, test_labels = test_labels)

datasets = np.load('./fashion_mnist.npz')
print(list(datasets.keys()))

train_images = datasets['train_images']
train_labels = datasets['train_labels']

test_images = datasets['test_images']
test_labels = datasets['test_labels']

print(train_images.shape)
print(train_labels.shape)

print(test_images.shape)
print(test_labels.shape)