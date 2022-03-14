from cmath import tanh
from random import randint,shuffle
from tkinter import W
from numpy import dtype, float32, matrix
import numpy as np
from numpy.random import rand
from sqlalchemy import column, null, true
import torch
from torch import gradient, nn, sigmoid
import math
import pandas as pd




#from torchvision import datasets
#from torchvision.transforms import ToTensor, Lambda, Compose
#import matplotlib.pyplot as plt


class activation_functions:
    def activate(self, val, code):
        if code == 0:
            return self.relu(val)
        elif code == 1:
            return self.sigmoid(val)
        elif code == 2:
            return self.tanh(val)
        elif code == 3:
            return self.softmax(val)
        return null
    
    def relu(self, vector):
        vector[vector < 0] = 0
        return vector
    def sigmoid(self, vector):
        return (1 / (1 + torch.exp(-1 * vector)))
    def tanh(self, vector):
        return (torch.exp(vector) - torch.exp(-1 * vector)) / (torch.exp(vector) + torch.exp(-1 * vector))
    def softmax(self, vector):
        return (torch.exp(vector)) / (torch.sum( torch.exp(vector) ))
    
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
        x = torch.zeros(11 , 7, dtype=torch.float32, requires_grad=True)
        with torch.no_grad():
            x.uniform_(-0.001, 0.001)

        
        ann_list.append(x)
        matrix = ann_list
        # init all the wrights by the xavier method
        #matrix = self.init_weights(ann_list, num_inputs)

        l = []
        for i in hl_nn:
            l.append(i[1])
        l.append(num_outputs[1])
        return matrix , l
    
    #foreward the model
    def foreward(self, vector_inputs):
        AF = activation_functions()
        calc = torch.matmul(vector_inputs, self.matrix[0])
        return calc
    
    def loss(self, y_pred, y):
        sum = torch.sum(torch.matmul(y , torch.log(y_pred))) * -1
        return sum
    
    def matrix_error(self, y_pred, y):
        return torch.argmax(y) == torch.argmax(y_pred)

        

        
    def train(self, dataset, lr):
        #prep data
        #loop
        AF = activation_functions()
        epoch_count = 1
        for epoch in dataset:
            count = 0
            correct = 0
            loss_total = 0
            #prep batch
            for batch in epoch:
                loss = torch.zeros(1)
                for sample in batch:
                    #foreward
                    y_pred = self.foreward(sample[0])
                    y_pred = AF.softmax(y_pred)
                    #calc loss
                    loss += self.loss(y_pred, sample[1])
                    loss_total += loss.item()

                    with torch.no_grad():
                        count += 1
                        if(self.matrix_error(y_pred, sample[1])):
                            correct += 1  
                #backward
                loss /= len(epoch[0])
                loss.backward()
                with torch.no_grad():
                    self.matrix[0] -= lr * self.matrix[0].grad
                #print("loss: ", loss)

                self.matrix[0].grad.zero_()
                  
            print(f'epoch {epoch_count}, accuracy: {correct/ count}, average loss: {loss_total / (len(epoch) * len(epoch[0]))}')
            epoch_count += 1
            
            
def split_dataset(x,y, batch_size, num_epochs):
    dataset = [] 
    ds_len = len(x) # get size of dataset
    batch_count = 0 
    batch = []   
    index = list(range(0, ds_len))
    for _ in range(num_epochs):
        epoch = []
        batch = []
        batch_count = 0
        shuffle(index)
        for i in index: # go over random list of all indexes in the
            if batch_count < batch_size:
                batch_count += 1
                batch.append( [x[i], y[i]] )
            else:
                epoch.append(batch)
                batch = []
                batch_count = 0

        dataset.append(epoch)
    return dataset  




def main():
    A = ANN([[7,0]], 11, [7, 1])
    df = pd.read_csv('winequality-white.csv')
    y = df.quality
    df.drop('quality', inplace=True, axis=1)
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(df)
    df = pd.DataFrame(sc.fit_transform(df), columns=df.columns)
    x = torch.tensor(df.values, dtype=torch.float32)
    y_val = y.values
    y = []
    print(A.matrix[0], "\n\n")
    for val in y_val:
        sample = [0] * 7
        sample[val - 3] = 1
        y.append(sample)
    y = torch.tensor(y, dtype=torch.float32)
    ds = split_dataset(x, y, 31, 150)
    A.train(ds, 0.001)
    
if __name__ == "__main__":
    main()