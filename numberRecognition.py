from src.config.tensorflowConfig import setup_tensorflow

tf = setup_tensorflow()

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

my_new_model = tf.keras.models.load_model('models/number_image_recognition')

new_val_loss, new_val_acc = my_new_model.evaluate(test_images, test_labels)
print('New Test accuracy', new_val_acc)
