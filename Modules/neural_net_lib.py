import math as m

class NeuralNet:
    def __init__(self,inp,hid,out,epsilon):
        #initial parameters are numbers of neurons in input (inp), hidden (hid), and output (out) layers 
        #plus epsilon value for weight initialization
        #firstly, write down numbers of neurons in each layer 
        self.inp=inp
        self.hid=hid
        self.out=out
        #secondly, create weights
        #wa - weights of connection between input and hidden layers
        #wb - weights of connection between hidden and output layers
        import random
        self.wa=[]
        self.wb=[]
        
        #fill weight tables with random values from -epsilon to epsilon
        for i in range(self.inp):
            weights_row=[] 
            for j in range(self.hid):
                weights_row.append(random.uniform(-epsilon,epsilon)) 
            self.wa.append(weights_row)  
            
        for i in range(self.hid):
            weights_row=[] 
            for j in range(self.out):
                weights_row.append(random.uniform(-epsilon,epsilon)) 
            self.wb.append(weights_row) 
            
    #define activation function
    def sigmoid(self,x):
        return 1/(1+m.exp(-x))
    
    #calculate y output
    def y(self,x):
        
        #firstly, calculate outputs of hidden layer neurons
        hidden_values=[]
        for j in range(self.hid):    
            yj=0
            for i in range(self.inp):
                yj+=x[i]*self.wa[i][j] #i - number of input layer neuron , j - number of hidden layer neuron

            yj=self.sigmoid(yj)
            hidden_values.append(yj)

        #secondly, hidden layer is treated as an input layer
        output_values=[]
        for k in range(self.out):
            zk=0
            for j in range(self.hid):
                zk+=hidden_values[j]*self.wb[j][k] #j - number of hidden layer neuron , k - number of output layer neuron

            zk=self.sigmoid(zk)
            output_values.append(zk)
            
        return output_values
    
    def train(self,x_data,y_data,eta):
        for t in range(len(x_data)):
            #firstly, calculate outputs of hidden layer neurons
            hidden_values=[]            
            for j in range(self.hid):    
                yj=0
                for i in range(self.inp):
                    yj+=x_data[t][i]*self.wa[i][j]#i - number of input layer neuron , j - number of hidden layer neuron

                yj=self.sigmoid(yj)
                hidden_values.append(yj)

            #secondly, hidden layer is treated as an input layer
            output_values=[]            
            for k in range(self.out):
                zk=0
                for j in range(self.hid):
                    zk+=hidden_values[j]*self.wb[j][k] #j - number of hidden layer neuron , k - number of output layer neuron

                zk=self.sigmoid(zk)
                output_values.append(zk)
            
            delta_k=[]
            for k in range(self.out):
                delta_k.append((output_values[k]-y_data[t][k])*output_values[k]*(1-output_values[k]))
                               
            delta_j=[]
            for j in range(self.hid):
                s=0
                for k in range(self.out):
                    s+=delta_k[k]*self.wb[j][k]
                delta_j.append(s*hidden_values[j]*(1-hidden_values[j]))
                               
            for j in range(self.hid):
                for k in range(self.out):
                    self.wb[j][k]-=eta*delta_k[k]*hidden_values[j]
                               
            for i in range(self.inp):
                for j in range(self.hid):
                    self.wa[i][j]-=eta*delta_j[j]*x_data[t][i]
