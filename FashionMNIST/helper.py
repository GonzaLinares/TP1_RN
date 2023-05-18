import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import albumentations as A

#classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def augmentData(train_X, train_y):
    
    #Buscar violinplot

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

    return train_X, test_X, train_y, test_y, test, labels, classes

def dataDistribution(plot):

    labels = pd.read_csv("./Data/train_labels.csv")
    train = np.load("./Data/train_images.npy")
    classes, frequency = np.unique(labels, return_counts=True)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    if plot == "classHist":
        plt.bar(classes, frequency)
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.title("Data Distribution")

    elif plot == "classMean":
        fig, axs = plt.subplots(2, 5)
        meanImg = [train[labels.label.values == k].mean(axis=0).reshape(28,28) for k in range(10)]
        for i in range(2):
            for j in range(5):
                axs[i][j].set_title(classes[5*i+j])
                axs[i][j].imshow(meanImg[5*i+j])

    elif plot == "intHist":
        fig, axs = plt.subplots(2, 5, figsize=(20,10))
        fig.tight_layout(pad=2)
        meanImg = [train[labels.label.values == k].mean(axis=0).reshape(28,28) for k in range(10)]
        for i in range(2):
            for j in range(5):
                axs[i][j].set_title(classes[5*i+j])
                lst = meanImg[5*i+j].flatten()
                axs[i][j].hist(lst)



def submit(model, test_X, y):

    predictions = model.predict(test_X)
    return predictions

if __name__ == "__main__":

    ...
