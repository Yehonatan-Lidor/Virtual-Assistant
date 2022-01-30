from cmath import tanh
from numpy import float32
import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader
#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt


class activation_functions:
    def relu(self, vector):
        vector[vector < 0] = 0
        return vector
    def sigmoid(self, vector):
        return (1 / (1 + torch.exp(-1 * vector)))
    def tanh(self, vector):
        return (torch.exp(vector) - torch.exp(-1 * vector)) / (torch.exp(vector) + torch.exp(-1 * vector))

class ANN:
    def create_ann(self, AF, hl_nn):
        """the hl_nn is a 2d array that represent how many hidden layers are there
        and how many neoros are inside ... each inner list has 2 values - the first one is 
        how many neorus are inside, and the second one is what activation function does it use
        """
        ann_list = list()
        for hl in hl_nn:
            ann_list.append(torch.zeros(hl[0], dtype=torch.float32))
        print(ann_list)

def main():
    AF = activation_functions()
    A = ANN()
    print(A.create_ann(AF, [[2,4], [8,6], [7,12], [5,14]]))

if __name__ == "__main__":
    main()

