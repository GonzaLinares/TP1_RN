import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as A
from tensorflow.keras.models import load_model

def augmentData(train_X, train_y):

    transform = A.Compose([A.VerticalFlip(always_apply=True),
                           A.HorizontalFlip(always_apply=True)])
    transformed = []

    transformed.append(transform(image=train_X))

    augmentedX = np.concatenate([train_X, np.flip(transformed[0]["image"])])
    augmentedy = np.concatenate([train_y, train_y])

    return augmentedX, augmentedy 

def initData():

    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    test = np.load("./Data/test_images.npy")
    test = test/255

    train = np.load("./Data/train_images.npy")
    train = train/255

    labels = pd.read_csv("./Data/train_labels.csv")

    train_X = train[:50000]
    test_X = train[50000:]
    train_y = labels[:50000].to_numpy()
    test_y = labels[50000:].to_numpy()

    return train_X, test_X, train_y, test_y, train, test, labels, classes

def submit(modelPath, test):

    model = load_model(modelPath, compile=False)
    toSubmit = pd.DataFrame(columns=["id","Category"])
    toSubmit["id"] = np.arange(len(test))
    toSubmit["Category"] = np.argmax(model.predict(test), (1))
    toSubmit.to_csv("./Submit/sub1.csv", index=False)

if __name__ == "__main__":

    ...
