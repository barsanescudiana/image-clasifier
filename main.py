import imghdr

import cv2
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.python.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.python.keras.models import load_model


def setup():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # data_dir = 'data'
    # image_exts = ['jpeg', 'jpg', 'bmp', 'png']

    # Remove dodgy images
    # for image_class in os.listdir(data_dir):
    #     for image in os.listdir(os.path.join(data_dir, image_class)):
    #         image_path = os.path.join(data_dir, image_class, image)
    #         try:
    #             # checking the image can load in opencv
    #             img = cv2.imread(image_path)
    #             # getting the image extension to check if it is in the ones listed above
    #             tip = imghdr.what(image_path)
    #             if tip not in image_exts:
    #                 print('Image not in ext list {}'.format(image_path))
    #                 os.remove(image_path)
    #         except Exception as e:
    #             print('Issue with image {}'.format(image_path))
                # os.remove(image_path)
    # Load data
    # Creating data pipeline using tensorflow
    data = tf.keras.utils.image_dataset_from_directory('data')
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(10, 10))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    # Preproccesing
    # Scale data
    data = data.map(lambda x, y: (x / 255, y))
    data.as_numpy_iterator().next()

    # Split data
    train_size = int(len(data) * .7)
    val_size = int(len(data) * .2)
    test_size = int(len(data) * .1)

    # Establishing the train, validation and test batches taken
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size).take(test_size)

    # Building model
    model = Sequential()
    model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
    # Condensing the value of the images
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(16, (3, 3), 1, activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    # Final layer that outputs if the image in Sad (1) or Happy (0)
    model.add(Dense(1, activation='sigmoid'))

    # Using the Adam optimizer
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    model.summary()

    logdir = 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # Plotting performance
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    # Evaluating model performance
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
    # model.save(os.path.join('models','imageClassifierModel.h5'))

def test(file_path):
    img = cv2.imread(file_path)
    # changing the color mode to show the image with RGB colors
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Resizing the image so it is 256x256 px
    resize = tf.image.resize(img, (256, 256))

    # Loading the model created in the setup()
    model = load_model(os.path.join('models', 'imageClassifierModel.h5'))

    # Classifying image using the model created earlier
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    print(yhat)

    if yhat > 0.5:
        print('Predicted class is Sad')
    else:
        print('Predicted class is Happy')

if __name__ == '__main__':
    # setup()
    test('happy-person-test-2.jpg')
