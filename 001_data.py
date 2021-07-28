import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

'''
Data
'''

# download training data
train = datasets.MNIST("", train=True, download=True, 
                        transform = transforms.Compose([transforms.ToTensor()]))

# download testing data
test = datasets.MNIST("", train=False, download=True, 
                        transform = transforms.Compose([transforms.ToTensor()]))

'''
batch: how many at the time to be passed to the model
shuffle: whether to shuffle the data before each epoch
'''
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# explore the data
for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]

print(y)

# check shape
print(data[0][0].shape)


plt.imshow(data[0][0].view(28, 28))
plt.show()




