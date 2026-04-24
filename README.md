# The-Self-Pruning-Neural-Network
This repository contains the implementation of a custom "self-pruning" neural network trained on the CIFAR-10 dataset. Instead of using post-training pruning techniques, this network dynamically learns to identify and remove its own weakest connections during the training loop using learnable gate parameters and L1 regularization.
