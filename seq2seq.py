from cmath import tanh
from random import randint, shuffle
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


def split_dataset(x,y, batch_size, num_epochs):
    dataset = [] 
    ds_len = len(x) # get size of dataset
    batch_count = 0 
    batch = []   
    for _ in range(num_epochs):
        epoch = []
        batch = []
        batch_size = 0
        for i in shuffle(list(range(0, ds_len))): # go over random list of all indexes in the
            if batch_count < batch_size:
                batch_size += 1
                batch.append( [x[i], y[i]] )
            else:
                epoch.append(batch)
                batch = []
                batch_size = 0
        dataset.append(epoch)
    return dataset
        
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
<<<<<<< HEAD
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

=======
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
>>>>>>> 35fe4bffd882a49526469ea8be846cb5f856d695
        # init all the wrights by the xavier method
        matrix = self.init_weights(ann_list, num_inputs)

        l = []
        for i in hl_nn:
            l.append(i[1])
        l.append(num_outputs[1])

<<<<<<< HEAD
        print(matrix)
=======
>>>>>>> 35fe4bffd882a49526469ea8be846cb5f856d695
        return matrix , l
    
    #foreward the model
    def foreward(self, vector_inputs):
<<<<<<< HEAD
=======
        AF = activation_functions()
<<<<<<< HEAD
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
        return AF.softmax(calc)
    
    def loss(self, y_pred, y):
        sum = torch.sum(torch.matmul(y,torch.log(y_pred))) * -1
        return sum
    
    def backward(self, x, y):
        lr = 0.1
        y_pred = self.foreward(x)
        l = self.loss(y_pred, y)
        l.backward()
        with torch.no_grad():
            self.matrix[0] -= lr * self.matrix[0].grad
        self.matrix[0].grad_zero_() 
        return y_pred
        
    def train(self, x, y, n_iter, batch_size):
        for epoch, X, Y in range(n_iter) , x, y:
            y_pred = self.backward(X, Y) # back propogate once
            print(f'epoch {epoch+1},  prediction:{y_pred}')
            
    
    


def main():
    
    
    
=======
>>>>>>> f678b7b5ad26eca9401944a1c15c8bf6c386e290
        calc = torch.matmul(vector_inputs, self.matrix)
        return calc
    def loss(self, y_pred, y):
        sum = torch.sum(torch.matmul(y , torch.log(y_pred))) * -1
        return sum
    def matrix_error(self, y_pred, y):
        return torch.argmax(y) == torch.argmax(y_pred)

        

        
    def train(self, epoch, lr, y):
            #prep data
            #loop
            for batch in epoch:
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
            print(f'accuracy: {correct/ count}')


class RNN(ANN):
    def __init__(self, num_inputs, num_output):
        super().__init__([[]], num_inputs + num_output[0], num_output)
    def train(self, epoch, lr, inputs, y , batch):
        first_input_addition = torch.randn(self.num_outputs, dtype=torch.float32)
        for i in range(epoch):
                count = 0
                correct = 0
                #prep batch
                for sample, y_res in zip(batch):
                    #foreward
                    #split to word
                    words_samples = sample.split(' ')
                    #convert to vector using word of vector
                    ### TO DO
                    y_pred = []
                    temp = first_input_addition
                    for index in range(len(words_samples)):
                        run_sample = torch.cat([y_pred[index],temp]) # conctrate the vectors
                        temp = self.foreward(run_sample) #run the model and save it for the next run
                        y_pred.append(temp) # add it to the prediction
                    ##transform the list of y_pred to a single vector for the loss computation
                    ####TO DO
                    ##transform the y_res to a vector
                    #calc loss 
                    loss = self.loss(y_pred, y_res)
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


class LSTM:
    def __init__(self, num_inputs, num_outputs, size_cell=64):
        self.num_output = num_outputs
        self.cell = torch.zeros(size_cell)
        self.forget_gate = ANN([[]], num_inputs + num_outputs, [size_cell, 10])
        self.add_gate_classify = ANN([[]], num_inputs + num_outputs, [size_cell, 10])
        self.add_gate_data = ANN([[]], num_inputs + num_outputs, [size_cell, 10])
        self.gate_out = ANN([[]], num_inputs + num_outputs, [num_outputs, 10])
        self.cell_to_out = ANN([[]], size_cell, [num_outputs, 10])
    def foreward(self, last_output, vector_inputs):
        AF = activation_functions()
        #concerate
        final_input_vector = torch.cat(vector_inputs, last_output)

        #step 1: forget gatגק
        #pass it through the gorget gate and apply sigmoid on the result
        forget_vector = self.forget_gate.foreward(final_input_vector)
        #apply sigmoid on the result
        forget_vector = AF.sigmoid(forget_vector)
        #multuply it by the cell
        self.cell = torch.matmul(self.cell, forget_vector)

        #step 2 - adding to cell
        #chose which data to add - multiply ANN and apply sigmoid
        adding_classify = self.add_gate_classify.foreward(final_input_vector)
        adding_classify = AF.sigmoid(adding_classify)
        #process the data with tanh
        new_cell_data = self.add_gate_data.foreward(final_input_vector)
        new_cell_data = AF.tanh(new_cell_data)

        #mutiply the vectors and it to the cell
        new_cell_data = torch.matmul(new_cell_data, adding_classify)
        self.cell = self.cell + new_cell_data

        #step 3 - process the vector and the cell and return the result
        #process the cell
        process_cell = self.cell_to_out.foreward(self.cell)
        process_cell = AF.tanh(process_cell)
        process_input_vector = self.gate_out.foreward(final_input_vector)
        process_input_vector = AF.sigmoid(process_input_vector)
        #multiply them
        ret_value = torch.matmul(process_cell, process_input_vector)
        return ret_value


    def train(self, dataset, lr):
        #prep data
        static_sample = torch.rand(self.num_output)
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
                temp = static_sample
                for sample in batch:
                    #foreward
                    final_vector = torch.concat(temp, sample[0])
                    temp = self.foreward(final_vector)
                    temp = AF.softmax(temp)
                    #calc loss
                    loss += self.loss(temp, sample[1])
                    loss_total += loss.item()

                    with torch.no_grad():
                        count += 1
                        if(self.matrix_error(temp, sample[1])):
                            correct += 1  
                #backward
                loss /= len(epoch[0])
                loss.backward()
                with torch.no_grad():
                    self.forget_gate.matrix[0] -= lr * self.forget_gate.matrix[0].grad
                    self.add_gate_classify.matrix[0] -= lr * self.forget_gate.matrix[0].grad
                    self.add_gate_data.matrix[0] -= lr * self.forget_gate.matrix[0].grad
                    self.cell_to_out.matrix[0] -= lr * self.forget_gate.matrix[0].grad
                    self.gate_out.matrix[0] -= lr * self.forget_gate.matrix[0].grad
                #print("loss: ", loss)

                self.matrix[0].grad.zero_()
                  
            print(f'epoch {epoch_count}, accuracy: {correct/ count}, average loss: {loss_total / (len(epoch) * len(epoch[0]))}')
            epoch_count += 1

 
        



        







def main():
    x = torch.tensor([1,2,3])
    y = torch.tensor([3,2,1])
    print(x + y)            
        
        

>>>>>>> 35fe4bffd882a49526469ea8be846cb5f856d695

if __name__ == "__main__":
    main()

