import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


'''
Using a more realistic dataset than in previous exercise;
requires some preprocessing
'''


'''
for pre-processing we don't want to build the data every time
in case of large datasets which can often take days,
in such case we can often pre-process the data once
'''

# flag:
REBUILD_DATA = False

class DogsVSCats():
    # 50px image size 
    # probably not a good idea as it will distort the image
    # but its ok for the purpose of this exercice
    # we could do padding to make it a square as a lazy option
    SIZE = 50
    CATS = "PetImages/Cat"
    DOGS = "PetImages/Dog"
    LABELS = {CATS: 0, DOGS: 1}

    training_data = []
    catcount = 0
    dogcount = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                try:
                    # file paths
                    path = os.path.join(label, f)
                    # read image and convert to grayscale
                    # for cats/dogs color is not a relavent feature
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    # resize image
                    img = cv2.resize(img, (self.SIZE, self.SIZE))
                    # add to training data, convert using one hot encoding
                    self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                    # count the number of cats/dogs 
                    # (sample balance is impornant)
                    if label == self.CATS:
                        self.catcount += 1
                    else:
                        self.dogcount += 1
                except Exception as e:
                    #print(e)
                    pass
        # shuffle the data
        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print("Saved training data to file")
        print("Number of cats: {}".format(self.catcount))
        print("Number of dogs: {}".format(self.dogcount))

if REBUILD_DATA:
    d = DogsVSCats()
    d.make_training_data()


# load the training data
training_data = np.load("training_data.npy", allow_pickle=True)
# check if training data is loaded
print(len(training_data))

# print the first image
print(training_data[1])

# show the image
plt.imshow(training_data[1][0], cmap='gray')
plt.show()

                





