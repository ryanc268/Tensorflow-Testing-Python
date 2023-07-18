import numpy as np
from matplotlib import pyplot as plt
import os
from src.config.tensorflowConfig import setup_tensorflow


tf = setup_tensorflow()

test_dir = 'resources/images/test/fruits'
print(os.listdir(test_dir))

model = tf.keras.models.load_model('models/image_recognition')
fruits = {0: "apple", 1: "banana", 2: "blueberry", 3: "mango",
          4: "orange", 5: "pineapple", 6: "raspberry", 7: "strawberry"}

for fruit in os.listdir(test_dir):
    img = tf.keras.preprocessing.image.load_img(os.path.join(test_dir, fruit), target_size=(256, 256))
    img_data = tf.keras.preprocessing.image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    prediction = model.predict(img_data)
    prediction_num = int(np.argmax(prediction))
    print(prediction)
    print(prediction_num)
    print('Prediction: {} - File: {}'.format(fruits[prediction_num], fruit))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
