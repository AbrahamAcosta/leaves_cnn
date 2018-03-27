# Abraham Acosta, CS410-Vision

# implement a neural network that classifies photos of elm, oak, and maple leaves.

# images are from the Leafsnap Dataset, from the paper:
#     "Leafsnap: A Computer Vision System for Automatic Plant Species Identification,"
#     Kumar, Belhumeur, Biswas, Jacobs, Kress, Lopez, Soarez,
#     Proceedings of the 12th European Conference on Computer Vision,
#     October 2012

import numpy
import cv2
import torch
import torchvision
import matplotlib.pyplot
import tqdm

dblTrain = []
dblValidation = []

# creating a data loader for the training samples of the leaves dataset
# leafsnap's lab images for maples, oaks, and elms will serve as the training set
# specifying the batch size and making sure it runs in a background thread

leafTrain = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=True,
	num_workers=1,
	pin_memory=False,
	dataset=torchvision.datasets.ImageFolder(
		root='eom/train',
		transform=torchvision.transforms.Compose([
			torchvision.transforms.RandomRotation(180),
			torchvision.transforms.Resize((28,28)),
			torchvision.transforms.Grayscale(1),
			torchvision.transforms.ToTensor()
		])
	)
)

# creating a data loader for the validation samples of the fashion dataset

leafValidate = torch.utils.data.DataLoader(
	batch_size=64,
	shuffle=True,
	num_workers=1,
	pin_memory=False,
	dataset=torchvision.datasets.ImageFolder(
		root='eom/validate',
		transform=torchvision.transforms.Compose([
			torchvision.transforms.RandomRotation(180),
			torchvision.transforms.Resize((28,28)),
			torchvision.transforms.Grayscale(1),
			torchvision.transforms.ToTensor()
		])
	)
)

# defining the network
# the network and its layers are summarized in the table below
# layers that were being experimented with are commented out

class Network(torch.nn.Module):

	def __init__(self):
	    super(Network, self).__init__()
	    self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=5)
	    # self.norm1 = torch.nn.BatchNorm2d(64)
	    self.conv2 = torch.nn.Conv2d(64, 512, kernel_size=5)
	    # self.norm2 = torch.nn.BatchNorm2d(512)
	    self.fc1 = torch.nn.Linear(2048, 256)
	    self.fc2 = torch.nn.Linear(256, 128)
	    self.fc3 = torch.nn.Linear(128, 10)

	# end

	def forward(self, x):
	    x = self.conv1(x)
	    # x = self.norm1(x)
	    x = torch.nn.functional.relu(x)
	    x = torch.nn.functional.max_pool2d(x, kernel_size=3)
	    x = self.conv2(x)
	    # x = self.norm2(x)
	    x = torch.nn.functional.relu(x)
	    x = torch.nn.functional.max_pool2d(x, kernel_size=2)
	    x = x.view(-1, 2048)
	    x = self.fc1(x)
	    # x = torch.nn.functional.dropout(x, p=0.35, training=self.training)
	    x = torch.nn.functional.relu(x)
	    x = self.fc2(x)
	    # x = torch.nn.functional.dropout(x, p=0.35, training=self.training)
	    x = torch.nn.functional.relu(x)
	    x = self.fc3(x)

	    return torch.nn.functional.log_softmax(x, dim=1)
	# end
# end

moduleNetwork = Network()

# specifying the optimizer based on adaptive moment estimation, adam
# it will be responsible for updating the parameters of the network

leafOptimizer = torch.optim.Adam(params=moduleNetwork.parameters(), lr=0.001)

def train():
	# setting the network to the training mode, some modules behave differently during training

	moduleNetwork.train()

	# obtain samples and their ground truth from the training dataset, one minibatch at a time

	for tensorInput, tensorTarget in tqdm.tqdm(leafTrain):
		# wrapping the loaded tensors into variables, allowing them to have gradients
		# in the future, pytorch will combine tensors and variables into one type
		# the variables are set to be not volatile such that they retain their history

		variableInput = torch.autograd.Variable(data=tensorInput, volatile=False)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=False)

		# setting all previously computed gradients to zero, we will compute new ones

		leafOptimizer.zero_grad()

		# performing a forward pass through the network while retaining a computational graph

		variableEstimate = moduleNetwork(variableInput)

		# computing the loss according to the cross-entropy / negative log likelihood
		# the backprop is done in the subsequent step such that multiple losses can be combined

		variableLoss = torch.nn.functional.nll_loss(input=variableEstimate, target=variableTarget)

		variableLoss.backward()

		# calling the optimizer, allowing it to update the weights according to the gradients

		leafOptimizer.step()
	# end
# end

def evaluate():
	# setting the network to the evaluation mode, some modules behave differently during evaluation

	moduleNetwork.eval()

	# defining two variables that will count the number of correct classifications

	intTrain = 0
	intValidation = 0

	# iterating over the training and the validation dataset to determine the accuracy
	# this is typically done one a subset of the samples in each set, unlike here
	# otherwise the time to evaluate the model would unnecessarily take too much time

	for tensorInput, tensorTarget in leafTrain:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)

		variableEstimate = moduleNetwork(variableInput)

		intTrain += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	for tensorInput, tensorTarget in leafValidate:
		variableInput = torch.autograd.Variable(data=tensorInput, volatile=True)
		variableTarget = torch.autograd.Variable(data=tensorTarget, volatile=True)

		variableEstimate = moduleNetwork(variableInput)

		intValidation += variableEstimate.data.max(dim=1, keepdim=False)[1].eq(variableTarget.data).sum()
	# end

	# determining the accuracy based on the number of correct predictions and the size of the dataset

	dblTrain.append(100.0 * intTrain / len(leafTrain.dataset))
	dblValidation.append(100.0 * intValidation / len(leafValidate.dataset))

	print('')
	print('train: ' + str(intTrain) + '/' + str(len(leafTrain.dataset)) + ' (' + str(dblTrain[-1]) + '%)')
	print('validation: ' + str(intValidation) + '/' + str(len(leafValidate.dataset)) + ' (' + str(dblValidation[-1]) + '%)')
	print('')
# end

# training the model for 100 epochs, one would typically save / checkpoint the model after each one

for intEpoch in range(100):
	train()
	evaluate()
# end

# plotting the learning curve according to the accuracies determined in the evaluation function
# note that this will not work if you are connected to the linux lab via ssh but no x forwarding

if False:
	matplotlib.pyplot.figure(figsize=(8.0, 5.0), dpi=150.0)
	matplotlib.pyplot.ylim(79.5, 100.5)
	matplotlib.pyplot.plot(dblTrain)
	matplotlib.pyplot.plot(dblValidation)
	matplotlib.pyplot.show()
# end
