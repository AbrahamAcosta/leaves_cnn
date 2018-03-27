# Leaves CNN
This is my implementation of a convolutional neural network for classification of images of leaves. The network was trained and tested on the included images of elm, oak, and maple leaves. The images used come from the LeafSnap dataset ( leafsnap.com/dataset ). The images were cropped and downscaled from their original format.

The network is implemented with PyTorch, using the ImageFolder data loader. This means that the classes for the neural network are determined by the names of the folders that the images are in, allowing for easy introduction of more classes. After training on the provided dataset for 100 epochs, the network has consistent successful guest rates of 95% on training images and 67% on validation images.
