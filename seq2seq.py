from cmath import tanh
from random import randint
from tkinter import W
from numpy import dtype, float32, matrix
import numpy as np
from numpy.random import rand
from sqlalchemy import column, true
import torch
from torch import gradient, nn, sigmoid
from torch.utils.data import DataLoader
import math
import pandas as pd




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
        with torch.no_grad():
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
        if len(hl_nn) != 0:
            for index, i in enumerate(hl_nn):
                if index == 0:
                    x = torch.zeros(num_inputs, i[0], dtype=torch.float32, requires_grad=True)
                    ann_list.append(x)
                    count = i[0]
                else:
                    x = torch.zeros(count, i[0], dtype=torch.float32, requires_grad=True)
                    ann_list.append(x)
                    count = i[0]
            x = torch.zeros(count, num_outputs[0], dtype=torch.float32, requires_grad=True)
            ann_list.append(x)
        else:
            ann_list = torch.zeros(num_inputs, num_outputs[0], dtype=torch.float32, requires_grad=True)
        # init all the wrights by the xavier method
        matrix = self.init_weights(ann_list, num_inputs)

        l = []
        for i in hl_nn:
            l.append(i[1])
        l.append(num_outputs[1])

        return matrix , l
    #foreward the model
    def foreward(self, vector_inputs):
        AF = activation_functions()
        calc = torch.matmul(vector_inputs, self.matrix)
        return AF.softmax(calc)
    def loss(self, y_pred, y):
        sum = torch.sum(torch.matmul(y , torch.log(y_pred))) * -1
        return sum
    def matrix_error(self, y_pred, y):
        return torch.argmax(y) == torch.argmax(y_pred)

        

        
    def train(self, epoch, lr, inputs, y,batch):
            #prep data
            #loop
            for i in range(epoch):
                count = 0
                correct = 0
                #prep batch
                for sample in batch:

                    #foreward
                    y_pred = self.foreward(sample)
                    #calc loss
                    loss = self.loss(y_pred, y)
                    #backward
                    loss.backward()
                    with torch.no_grad():
                        self.matrix -= lr * self.matrix.grad
                    self.matrix.grad.zero_()
                    #print epoch
                    with torch.no_grad():
                        count += 1
                        if(self.matrix_error(y_pred, y)):
                            correct += 1
            print(f'epoch {i+1}, accuracy: {correct/ count}')






def main():
    #load data
    df = pd.read_csv('features.csv')
    output = df.output
    df = df.drop(columns=["index", "query", "output"])    
    final_list = []
    for i in output:
        if i == "SEARCH":
            final_list.append(torch.tensor([1.0,0.0,0.0,0.0,0.0], dtype=torch.float32))
        elif i == "GET_MESSAGE":
            final_list.append(torch.tensor([0.0,1.0,0.0,0.0,0.0], dtype=torch.float32))
        elif i == "SEND_MESSAGE":
            final_list.append(torch.tensor([0.0,0.0,1.0,0.0,0.0], dtype=torch.float32))
        elif i == "GetWeather":
            final_list.append(torch.tensor([0.0,0.0,0.0,1.0,0.0], dtype=torch.float32))
        else:
            final_list.append(torch.tensor([0.0,0.0,0.0,0.0,1.0], dtype=torch.float32))
    print(len(df))
    print(len(final_list))
    final_db = []
    from sklearn.utils import shuffle
    epoch_list = []
    for _ in range(100):
        df = shuffle(df)
        count = 0
        epoch_list = []
        batch_list = []
        for sample, output in zip(df.itertuples(), final_list):
            count += 1
            batch_list.append([sample, output])
            if count % 64 == 0:
                count = 0
                epoch_list.append(batch_list)
                batch_list = []
        final_db.append(epoch_list)
        epoch_list = []
    print(final_db)
        
    


            
            






        


if __name__ == "__main__":
    main()

