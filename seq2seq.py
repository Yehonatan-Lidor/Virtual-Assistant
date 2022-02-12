from cmath import tanh
from random import randint
from tkinter import W
from numpy import dtype, float32, matrix
import numpy as np
from numpy.random import rand
import torch
from torch import gradient, nn, sigmoid
from torch.utils.data import DataLoader
import math

#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt


class activation_functions:
    def softmax(self, vector):
        return (torch.exp(vector)) / (torch.sum( torch.exp(vector) ))
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
        self.num_outputs = num_outputs
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
        print()
        print(ann_list)
        matrix = self.init_weights(ann_list, num_inputs)

        l = []
        for i in hl_nn:
            l.append(i[1])
        l.append(num_outputs[1])
        return matrix , l
    #foreward the model
    def foreward(self, vector_inputs):
        AF = activation_functions()
        calc = vector_inputs
        af = 18
        for index in range(len(self.matrix)):
            af = self.activations[index]
            calc = torch.matmul(calc , self.matrix[index])
            if af == 0:
                calc = AF.relu(calc)
            if af == 1:
                calc = AF.sigmoid(calc)
            if af == 2:
                calc = AF.tanh(torch.matmul(calc, self.matrix[index]))
        return calc
    def loss(self, y_pred, y):
        sum = torch.sum(y * torch.log(y_pred)) / self.num_outputs * -1
        return sum
    def train(self, lr, data, epochs, batch):
        for epoch in epochs:
            #split to batch
            x_train = []
            y_train = []            
            y_pred = []
            for sample in x_train:
                pass
                #foreward:
                y_pred.append(self.foreward(sample))    


            #calc loss
            loss = self.loss(y_pred, y_train)
            #calc gradient
            loss.backward()
            #update weights using back-prop



        



def main():
    X = torch.tensor([1], dtype=torch.float32)
    Y = torch.tensor([2], dtype=torch.float32)

    W = torch.tensor([100.0], dtype=torch.float32, requires_grad=True)

    # Training
    learning_rate = 0.01
    n_iters = 1000

    for epoch in range(n_iters):
        # predict = forward pass
        y_pred = W * X

        # loss
       # m = torch.nn.MSELoss(y_pred, Y)
        l = torch.sum(Y * torch.log(y_pred)) * -1
        # calculate gradients = backward pass
        l.backward()

        # update weights
        #w.data = w.data - learning_rate * w.grad
        with torch.no_grad():
            W -= learning_rate * W.grad
        
        # zero the gradients after updating
        W.grad.zero_()

        print(f'epoch {epoch+1}: w = {W.item():.3f}, loss = {l.item():.8f}, accuracy: {y_pred / Y * 100}')

        





    



        






if __name__ == "__main__":
    main()

