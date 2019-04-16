import numpy
import scipy.special
import scipy.misc
import scipy.misc.pilutil

class neuralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        # self.wih = (numpy.random.normal(self.onodes,self.inodes)-0.5)
        # self.who = (numpy.random.normal(self.onodes,self.hnodes)-0.5)
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self,inputs_list,targets_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T
        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_nodes = 784
hidden_nodes = 500
output_nodes = 10
learning_rate = 0.1
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
with open(r'C:\Users\Marc\PycharmProjects\untitled\NeuralNetwork\Database\mnist_train.csv','r')as tr:
    training_data_list = tr.readlines()
for record in training_data_list:
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:])/255.0 *0.99) + 0.01
    targets = numpy.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs,targets)

# with open(r'C:\Users\Marc\PycharmProjects\untitled\NeuralNetwork\Database\mnist_test_10.csv','r')as te:
#     test_data_file = te.readlines()
# all_values = test_data_file[0].split(',')
# print(all_values[0])
# n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01)
# print(n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01))

img_array = scipy.misc.imread(r'C:\Users\Marc\Desktop\yy.png',flatten = True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0*0.99)+0.01
n.query(img_data)
print(n.query(img_data))