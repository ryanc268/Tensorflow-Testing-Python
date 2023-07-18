import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from src.config.tensorflowConfig import setup_tensorflow


tf = setup_tensorflow()
train_dir = '../resources/images/train/fruits'
val_dir = '../resources/images/valid/fruits'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

print(os.listdir(train_dir))
# print('Cleaning Images...')
# # Remove bad images
# for fruit_type in os.listdir(train_dir):
#     for fruit_image in os.listdir(os.path.join(train_dir, fruit_type)):
#         image_path = os.path.join(train_dir, fruit_type, fruit_image)
#         try:
#             # check image can be read by opencv
#             img = cv2.imread(image_path)
#             # grab extension of image
#             ext = imghdr.what(image_path)
#             if ext not in image_exts:
#                 print('Image extension not supported: {} path: {}'.format(ext, image_path))
#                 os.remove(image_path)
#         except Exception as e:
#             print('issue with image {}'.format(image_path))
# print('Done Cleanup!')

# data = tf.keras.utils.image_dataset_from_directory(data_dir).map(lambda x, y: (x / 255, y))

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train = data_generator.flow_from_directory(train_dir, target_size=(256, 256), class_mode='categorical')
val = data_generator.flow_from_directory(val_dir, target_size=(256, 256), class_mode='categorical')

print('train length: {}'.format(len(train)))
print('val length: {}'.format(len(val)))

# 0 = Apple
# 1 = Banana
# 2 = Blueberry
# 3 = Mango
# 4 = Orange
# 5 = Pineapple
# 6 = Raspberry
# 7 = Strawberry

rows = 5
cols = int(len(train) / (rows - 1))
fig = plt.figure(figsize=(10, 10))
for idx in range(len(train)):
    fig.add_subplot(rows, cols, idx + 1)
    img, label = train.next()
    plt.imshow(img[0])
    plt.axis('off')
    plt.title(np.argmax(label[0]))

plt.show()

# print(len(data))
# train_size = int(len(data) * .7)
# val_size = int((len(data) - train_size) * .7)
# test_size = len(data) - train_size - val_size
# print(train_size, val_size, test_size)

# train = data.take(train_size)
# val = data.skip(train_size).take(val_size)
# test = data.skip(train_size + val_size).take(test_size)

model = keras.Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
# 8 Fruit folders = Dense down to 8 layers
model.add(Dense(8, activation='softmax'))

model.compile('adam', loss=tf.losses.CategoricalCrossentropy(), metrics=['accuracy'])
model.summary()

logdir = '../resources/logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback], verbose=1)

fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='loss')
plt.plot(history.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()

model.save('../models/image_recognition')
