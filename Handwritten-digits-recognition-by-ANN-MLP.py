import pandas as pd
import time, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import copy
from sklearn.metrics import mean_squared_error
import statistics
from scipy.stats import truncnorm

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)
activation_function = sigmoid

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


class NeuralNetwork:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.mse_arr = []
        self.learning_rate = learning_rate 
        self.create_weight_matrices()
        
    def create_weight_matrices(self):
        """ 
        A method to initialize the weight 
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, 
                                         self.no_of_hidden_nodes))
        
    
    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.wih, 
                                input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who, 
                                output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        # update the weights:
        tmp = output_errors * output_network \
              * (1.0 - output_network)     
        tmp = self.learning_rate  * np.dot(tmp, 
                                           output_hidden.T)
        self.who += tmp


        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, 
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * \
              (1.0 - output_hidden)
        self.wih += self.learning_rate \
                          * np.dot(tmp, input_vector.T)
        
        self.mse_arr.append(mean_squared_error(target_vector,output_network))
        
    
    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, 
                               input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, 
                               output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector
            
    def confusion_matrix(self, y_test, y_pred):
        y_test=list(y_test)
        cm = np.zeros((10, 10), int)
        for i in range(len(y_pred)):
            target = y_test[i]
            cm[y_pred[i], int(target)] += 1
        return cm    

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()
    
    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()
        
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
    
    
def convert_to_csv(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")
    
    f.read(16)
    l.read(8)
    images = []
    for i in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)
        
    for image in images:
        list = [str(pix) for pix in image]
        o.write(",".join(list)+"\n")
    f.close()
    o.close()
    l.close()

def svm_classifing(X_train,y_train,X_test,y_test,kernel_type):
    start_time = time.time()
    if kernel_type == 'poly':
        svclassifier = SVC(kernel=kernel_type, degree=8)
    else:
        svclassifier = SVC(kernel=kernel_type)                
    svclassifier.fit(X_train, y_train)
    
    y_pred = svclassifier.predict(X_test)
    print("*********************************************************************************")
    print("Traning time for SVM %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_train)), (time.time() - start_time)))
    print("*********************************************************************************")
    print("score = %3.2f" %svclassifier.score(X_test,y_test))
    
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    return y_pred

def show_samples(X_test, y_pred, rand_list, name):
    for i in rand_list:
        two_d = (np.reshape(X_test.values[i], (28, 28))).astype(np.uint8)
        plt.title('predicted label: {0}'. format(y_pred[i]))
        plt.imshow(two_d, interpolation='nearest', cmap='gray')
        plt.savefig('fig/'+name+'_'+str(i)+'.jpg')
        plt.show()

def show_mse(mse, name):
    plt.plot(mse)
    plt.title('Mean Square Error: '+name)
    plt.ylabel('Mean Square Error')
    plt.xlabel('iteration')
    plt.savefig('fig/'+'Mean Square Error_'+name+'.jpg')
    plt.show()
    
def show_digits_error(y_test, y_pred, name):
    y_test=list(y_test)
    correct_guess = [0] * 10
    incorrect_correct_guess = [0] * 10
    for i in range(len(y_test)):
        if y_test[i] != y_pred[i]:
            incorrect_correct_guess[y_test[i]] += 1
        else:
            correct_guess[y_test[i]] += 1
    
    print("accuracy: test", sum(correct_guess) / len(y_test))        
    
    N = 10
    fig, ax = plt.subplots()
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    
    p1 = ax.bar(ind, correct_guess, width)
    p2 = ax.bar(ind + width, incorrect_correct_guess, width)
    
    ax.set_title('Correct VS incorrect prediction: '+name)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    ax.legend((p1[0], p2[0]), ('Correct', 'Incorrect'))
    ax.autoscale_view()
    plt.savefig('fig/'+'Correct VS incorrect_'+name+'.jpg')
    plt.show()   

    percentage_err = [round(j*100 / (i+j), 2) for i, j in zip(correct_guess, incorrect_correct_guess)]    
    plt.xticks( range(len(percentage_err)) ) # location, labels
    plt.bar(np.arange(len(percentage_err)), percentage_err)
    plt.title('Percentage Error of Digits: '+name)
    plt.ylabel('Percentage Error')
    plt.xlabel('digit')
    plt.savefig('fig/'+'Percentage Error of Digits_'+name+'.jpg')
    plt.show()          
    
def ann_modeling(ANN,X_train, y_train, kernel_type, beta, epsilon, iteration, h_node):
    start_time = time.time()

    x = X_train.to_numpy()
    d = one_hot(y_train, 10)

    mse_tmp = 1
    mse = []
    while mse_tmp > epsilon:    
        # print("//////////////////////////////////////////////////////")
        # print("epoch: ", len(mse))
        for i in range(len(x)):
            ANN.train(x[i], d[i])
        mse_tmp = statistics.mean(ANN.mse_arr)
        mse.append(mse_tmp)        
        # print(mse_tmp)
        mse_tmp = np.mean(mse_tmp)
        ANN.mse_arr = []
        if len(mse) > iteration:
            break    

    print("*********************************************************************************")
    print("Traning time for ANN %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_train)), (time.time() - start_time)))
    print("Iteration: ", len(mse))
    print(mse_tmp)
    print("*********************************************************************************")
        
    return ANN, mse

def ann_classifing(ANN,X_test,y_test,kernel_type):
    start_time = time.time()

    x = X_test.to_numpy()
    d = one_hot(y_test, 10)

    y_pred = []    
    for i in range(len(x)):
        res = ANN.run(x[i])
        y_pred.append(np.argmax(res))

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
    print("*********************************************************************************")
    print("Test time for ANN %s with length %s:--- %s seconds ---" %( kernel_type, (len(X_test)), (time.time() - start_time)))
    print("*********************************************************************************")
    
    return ANN, y_pred
   
def one_hot(labels, length):
    one_hot_labels = np.zeros((labels.size,length))
    one_hot_labels[np.arange(labels.size),labels] = 1
    return one_hot_labels
   
def normalize(df):
    # return df.div(255)
    return df * 0.99/255 + 0.01

def data_selection(df, train_size, train_start, test_size, test_start, binary_threshold = 150):
    df1 = copy.deepcopy(df)
    X_train = df1.iloc[train_start:train_start+train_size,1:]
    if binary_threshold >=0:
        X_train[X_train <= binary_threshold] = 0
        X_train[X_train > binary_threshold] = 1
    else:
        X_train = normalize(X_train)
    
    X_test = df1.iloc[test_start:test_start+test_size,1:]
    if binary_threshold >=0:
        X_test[X_test <= binary_threshold] = 0
        X_test[X_test > binary_threshold] = 1
    else:
        X_test = normalize(X_test)

    y_train = df1.iloc[train_start:train_start+train_size, 0]
    y_test = df1.iloc[test_start:test_start+test_size, 0]
    
    return X_train, y_train, X_test, y_test
    
def ann_svm(df, train_size, train_start, test_size, test_start, binary_threshold, training_rate, epsilon, iteration, hidden_node, kernel_type):
    X_train, y_train, X_test,y_test = data_selection(df, train_size, train_start, test_size, test_start, binary_threshold)
    sns.countplot(y_train)
    plt.savefig('fig/'+'digit_bias_'+str(train_size)+'.jpg')
    plt.show()
    y_pred = svm_classifing(X_train,y_train,X_test,y_test,'linear')
    for beta in training_rate:
        for h_node in hidden_node:
            ANN = NeuralNetwork(no_of_in_nodes = 784, 
                        no_of_out_nodes = 10, 
                        no_of_hidden_nodes = h_node,
                        learning_rate = beta)
            name = str(kernel_type)+' - train size ' +str(train_size)+' - beta '+ str(beta)+' - hidden_node '+ str(h_node)+' - binary_threshold '+ str(binary_threshold)+' - epsilon '+ str(epsilon)+' - iteration '+ str(iteration)
            print('*********************************************************************************')
            print(name)
            ANN, mse = ann_modeling(ANN,X_train, y_train, kernel_type, beta, epsilon, iteration, h_node)
            ANN, y_pred = ann_classifing(ANN, X_test,y_test, kernel_type)
            show_mse(mse, name)
            print("mse_traning = ", statistics.mean(mse))
            show_digits_error(y_test, y_pred, name)
            cm = ANN.confusion_matrix(y_test, y_pred)
            print(cm)
            for i in range(10):
                print("digit: ", i, "precision: ", ANN.precision(i, cm), "recall: ", ANN.recall(i, cm))
    return X_test, y_pred 

#*******************************main******************************
# sys.stdout = open("log.txt", "w")
start_time = time.time()     
convert_to_csv("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
        "mnist_handwritten.csv", 22000)

rand_list = (np.random.randint(0,100,6))

df = pd.read_csv('mnist_handwritten.csv', sep=',', header=None)    
print("converting time: --- %s seconds ---" % (time.time() - start_time))
print(df.head(3))
print(df.shape)

sns.countplot(df[0])
plt.show()

training_rate = [0.9, 0.7, 0.5, 0.2, 0.01]
hidden_node = [10, 35, 100, 300, 500]
epsilon = 0.1
iteration = 300
binary_threshold = -1 # -1 for ignoring binary thresholding

X_test, y_pred = ann_svm(df, 10000, 0, 1000, 20000, binary_threshold, training_rate, epsilon, iteration, hidden_node, 'sigmoid')
show_samples(X_test, y_pred, rand_list, 'sigmoid - train size 10000 - beta 0.01')


    
print("total time: --- %s seconds ---" % (time.time() - start_time))    
# sys.stdout.close()