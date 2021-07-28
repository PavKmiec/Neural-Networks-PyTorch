import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

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


'''
# check the data
for data in trainset:
    print(data)
    break

x, y = data[0][0], data[1][0]
print(y)

# check shape
print(data[0][0].shape)

plt.imshow(data[0][0].view(28, 28))
plt.show()

to run intitialization of inherited class
we need to use super()
'''

# network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        Fully connected layer: fc
        Linear: 
            Applies a linear transformation to the incoming data: y=xA^T+b

        Parameters: (input, output)
         input (Tensor/Array): 
           in our case, an image of size 28x28px (rows of pixels)
           input is flattened to a vector/1d array of size 784
         output: the number of output neurons
        '''
         
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64) 
        self.fc3 = nn.Linear(64, 64)
        # output layer, (input, classes)
        self.fc4 = nn.Linear(64, 10)

    # forward pass
    def forward(self, x):
        # pass the input through the layers
        # use a relu activation function (F.relu)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        # return probability of each class 
        # x is the output of the network
        # dim=1 is the dimension along which the softmax is applied 
        return F.log_softmax(x, dim=1)

# sample data
X = torch.randn(28*28)
# shape dimension
X = X.view(-1, 28*28)

net = Net()

# output of the network
output = net(X)
print(output)


# print(net)





