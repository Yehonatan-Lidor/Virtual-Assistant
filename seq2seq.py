from cmath import tanh
from random import randint
from numpy import float32, matrix
import numpy as np
from numpy.random import rand
import torch
from torch import nn, sigmoid
from torch.utils.data import DataLoader
import math

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
    def __init__(self, hl_nn, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_inputs = num_outputs
        s = self.create_ann(hl_nn=hl_nn, num_inputs=num_inputs, num_outputs=num_outputs)
        self.matrix = s[0]
        self.activations = s[1]
        
    def init_weights(self, matrix, num_inputs):
        #init each vector of weights
        for inx, i  in enumerate(matrix):
            if inx == 0:
                lower, upper = (-1 * math.sqrt(6) / math.sqrt(num_inputs + len(matrix[inx]))), (math.sqrt(6) / math.sqrt(num_inputs + len(matrix[inx])))
            else:
                lower, upper = (-1 * math.sqrt(6) / math.sqrt(len(matrix[inx -1]) + len(matrix[inx]))), (math.sqrt(6) / math.sqrt(len(matrix[inx -1]) + len(matrix[inx])))
            count = 0
            while(count < len(i)):
                rand_number = rand(1)
                weight = lower + (upper - lower) * rand_number
                matrix[inx][count] = weight[0]
                count += 1
        return matrix
            
    def create_ann(self, hl_nn, num_inputs, num_outputs):
        """the hl_nn is a 2d array that represent how many hidden layers are there
        and how many neoros are inside ... each inner list has 2 values - the first one is 
        how many neorus are inside, and the second one is what activation function does it use
        """

        ##crate a represntation of the hidden layers - with list of tensors
        ann_list = list()
        count = 0
        for index, i in enumerate(hl_nn):
            if index == 0:
                x = torch.zeros(num_inputs, i[0])
                print(x[0])
                ann_list.append(x)
                print(i[0])
                count = i[0]
            else:
                x = torch.zeros(count, i[0])
                ann_list.append(x)
                count = i[0]
            print("\n")
        x = torch.zeros(count, num_outputs[0])
        ann_list.append(x)

        # init all the wrights by the xavier method
        
        matrix = self.init_weights(ann_list, num_inputs)

        l = []
        for i in hl_nn:
            l.append(i[1])
        l.append(num_outputs[1])
        return matrix , l
    #foreward the model
    def foreward(self, vector_inputs):
        print(vector_inputs)
        print(self.matrix)
        AF = activation_functions()
        calc = vector_inputs
        for index in range(len(self.matrix)):
            af = self.activations[index]
            af= 18
            calc = torch.matmul(calc , self.matrix[index])
            if af == 0:
                calc = AF.relu(calc)
            if af == 1:
                calc = AF.sigmoid(calc)
            if af == 2:
                calc = AF.tanh(torch.matmul(calc, self.matrix[index]))
        return calc
    def loss(self, y_pred, y):
        return torch.mean((y_pred - y)**2)
    def train(self, lr, data, epochs, batch):
        for epoch in epochs:
            #split to batch
            list_data = [] #list of all data splitted to the batch size
            for sample in list_data:
                pass
                #foreward:
                
                #calc loss

                #calc gradient

                #update weights using back-prop

        



def main():
    pass
    #print(a.shape)
    #print(b)
    #print(b.shape)
    #print(torch.mm(a,b))
    AF = activation_functions()
    A = ANN([[1, 0]], 1, [1, 0])
    a = torch.tensor([2], dtype=torch.float32)
    print("\n\n")
    print(A.foreward(a))
    #t = torch.randn((3,2))
    #print(t)
    #print("\n\n")
    #print(torch.transpose(t,1,0))
    """
    1 * 
    """



if __name__ == "__main__":
    main()

