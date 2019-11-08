from phe import paillier as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
import pandas as pd
import math
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import random
import time
import glob, os

# 画像をリサイズする
def resize(input_parent, input_fname, label):
    img = cv2.imread(input_parent + "/" + input_fname)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = cv2.resize(img, dsize=(100, 100))
    if not os.path.exists("./resize" + label):
        os.mkdir("./resize" + label)
    cv2.imwrite("./resize" + label + "/" + input_fname, dst)
    print("resize: ", input_fname)

# api 用
def api_resize(input_parent, input_fname):
    img = cv2.imread(input_parent + "/" + input_fname)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dst = cv2.resize(img, dsize=(100, 100))
    if not os.path.exists("./api_resize"):
        os.mkdir("./api_resize")
    cv2.imwrite("./api_resize/" + input_fname, dst)
    print("resize: ", input_fname)

def load_dataset():
    '''
    This function is used for loading and transforming human images dataset
    from scikit-learn package

    Returns:
        Four numpy arrays. The details of each array are follows:
            1.) First array contains all training samples.
            2.) Second array stores the labels corresponding with the first array.
            3.) Third array contains testing samples.
            4.) Fourth array stores the actual label of each testing sample.

    '''
    # load original image
    #X = [[random.random() for i in range(0, 10000)] for i in range(0, 100)]
    #y = [random.randint(0, 1) for i in range(0, 100)]
    #X = np.array(X)
    #y = np.array(y)

    print("==================================")
    print("load images")

    img0_path_list = os.listdir("images0")
    for i in range(len(img0_path_list)):
        resize("images0", img0_path_list[i], '0')
    img1_path_list = os.listdir("images1")
    for i in range(len(img1_path_list)):
        resize("images1", img1_path_list[i], '1')

    print("===================================")
    print("reshape")
    #FACE_CASCADE_PATH = ('C:\\Users\\wakame\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
    #face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

    in_dim = 26  # 入力ベクトルの次元    
    re0_path_list = os.listdir("resize0")
    #X = np.empty((0, 10000), int)
    X = np.empty((0, in_dim), int)
    y = np.empty((0, 1), int)
    for i in range(len(re0_path_list)):
        img = cv2.imread("resize0/" + re0_path_list[i])
        img = np.array(img).reshape(10000, 3).sum(axis=1) / 3
        img = img.reshape(10000, )
        # 10刻みで輝度値を取得するためのベクトル(26 dim)
        tmp_vec = np.zeros(26)
        for j in range(img.shape[0]):
            tmp_vec[int(img[j]/10)] = tmp_vec[int(img[j]/10)] + 1
        #X = np.append(X, [img], axis=0)
        X = np.append(X, [tmp_vec], axis=0)
        y = np.append(y, 0)
    re1_path_list = os.listdir("resize1")
    for i in range(len(re1_path_list)):
        img = cv2.imread("resize1/" + re1_path_list[i])
        img = np.array(img).reshape(10000, 3).sum(axis=1) / 3
        img = img.reshape(10000, )
        tmp_vec = np.zeros(26)
        for j in range(img.shape[0]):
            tmp_vec[int(img[j]/10)] = tmp_vec[int(img[j]/10)] + 1
        #X = np.append(X, [img], axis=0)
        X = np.append(X, [tmp_vec], axis=0)
        y = np.append(y, 1)
    
    #print(X)
    print(X[:1])
    print(X.shape)
    #print(y)
    print(y[:2])
    print(y.shape)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_max = X.max(axis=0)
    X_min = X.min(axis=0)

    print(X[:1])

    #x_train = np.concatenate((X[:5], X[50:55]))
    #y_train = np.concatenate((y[:5], y[50:55]))

    th = 3
    dsize = X.shape[0]

    x_train = X[th:dsize-th]
    y_train = y[th:dsize-th]
    x_test = np.concatenate((X[:th], X[dsize-th:dsize]))
    y_test = np.concatenate((y[:th], y[dsize-th:dsize]))

    print("training datasets: ", x_train.shape)
    print("test datasets: ", x_test.shape)

    return x_train, x_test, y_train, y_test, X_max, X_min

def calculate_accuracy(actual_label, prediction_label):
    '''
    An utility function for calculating the accuracy of the model.

    Args:
        actual_labels: a list that stores actual label of every sample in the testing dataset.
        prediction_labels: a list that stores prediction label outputed from the model.

    Returns: 
        accuracy in string format.
    '''
    correct = 0
    for i, prediction in enumerate(prediction_label):
        if actual_label[i] == prediction:
            correct+=1

    return '{0:.2f}%'.format(100.*correct/prediction_label.size)


def simple_sigmoid(input):
    '''
    A simple implementation of sigmoid function.

    Args:
        input: float value.
    
    Returns:
        sigmoid value of input.
    '''

    if input < 0:
        return 1 - 1 / (1 + math.exp(input))
    return 1/(1+math.exp(-input))


def make_simple_logistic_prediction(encrypted_features, weights):
    '''
    This function predicts the class of a encrypted testing sample.

    Args:
        encrypted_features:  a numpy array that stores encrypted feature values of the input sample.
        weights: a numpy array that stores trained weights.
        
    Returns:
        An EncryptedNumber object that stores a prediction.
    '''
    return simple_sigmoid(np.dot(encrypted_features, weights))

def training_logistic_regression(training_samples, training_labels, epoch=3000, learning_rate=0.01, limited_sum_weights=3):
    '''
    A training function for logistic regression.

    Args:
        training_dataset: a numpy array that stores feature values of every sample in dataset.
        training_labels: a numpy array that stores class labels of every sample in dataset.
        epoch: number of epoch for training. Default value is 10.
        learning_rate: learning rate of logistic regression. Default value is 0.1.
        limited_sum_weights: The summation of the weights should not be higher than this limited_sum_weights threshold. 

    Returns:
        weights
    '''
    
    # initializing weights
    weights = np.zeros(training_samples.shape[1])
    for i in range(epoch):
        cum_grad = np.zeros(training_samples.shape[1])
        tmp_weights = np.array(weights)
        for idx, sample in enumerate(training_samples):
            label = training_labels[idx]
            prediction = make_simple_logistic_prediction(sample, weights)
            grad = np.dot(sample, prediction-label)
            cum_grad += grad
        cum_grad = learning_rate*cum_grad/len(training_samples)
        weights -= cum_grad
        
        #our stopping criteria
        if abs(np.sum(weights)) > limited_sum_weights:
            return tmp_weights
        
    return weights

##################################


def encrypt_logistic_regression_dataset(public_key:pl.PaillierPublicKey, dataset):
    '''
    This functions is used for encrypting dataset using Paillier.

    Args:
        public_key: PaillierPublicKey object.
        dataset: a numpy array that stores feature values of every sample in the dataset.
        
    Returns:
        Numpy array that stores the encrypted dataset.
    '''
    encrypted_dataset = np.empty(dataset.shape, dtype=object)
    for row in range(dataset.shape[0]):
        for col in range(dataset.shape[1]):
            encrypted_dataset[row, col] = public_key.encrypt(dataset[row, col])

    return encrypted_dataset

def pre_prediction(dataset, trained_weights):
    '''
    A function to calculate dot product between features and trained_weights.

    Args:
        dataset: numpy array of encrypted feature values of every predicting samples.
        trained_weight: numpy array of weights that have been trained.

    Returns:
        dot product between dataset and trained_weights.
    '''
    return np.dot(dataset, trained_weights)

def calcualte_sigmoid_and_decrypt_result(private_key: pl.PaillierPrivateKey, pre_prediction_results):
    '''
    Function to produce the prediction output.

    Args:
        private_key: PaillierPrivateKey object.
        pre_prediction_results: numpy array obtained from pre_prediction function.

    Returns:
        Two numpy arrays. The first one stores sigmoid value of every predicting sample.
        The second one stores the prediction class label of each predicting sample.

    '''
    output = np.empty(pre_prediction_results.shape, dtype=float)
    output_class = np.empty(pre_prediction_results.shape, dtype=int)

    for i in range(pre_prediction_results.size):
        output[i] =  simple_sigmoid(private_key.decrypt(pre_prediction_results[i])) 
        output_class[i] = 1 if output[i] >= 0.5 else 0
    return output, output_class



def logistic_regression():
    '''
    This function shows example of training and running logistic regression implemented by using Paillier scheme.
    '''
    #getting keys for paillier
    public_key, private_key = pl.generate_paillier_keypair(n_length=256)
    x_train, x_test, y_train, y_test, X_max, X_min = load_dataset()

    print('=========== Training logistic regression ===========')
    weights = training_logistic_regression(x_train, y_train, epoch=3000, learning_rate=0.1)  #simple training function    
    print('Trained weights > ', weights)

    print('=========== Encrypted test data ===========')
    start = time.time()

    encrypted_x_test = encrypt_logistic_regression_dataset(public_key, x_test)

    encrypted_time = time.time() - start
    print("encrypt time:{0}".format(encrypted_time) + "[sec]")
    """
    for row in range(3):
        for col in range(encrypted_x_test.shape[1]):
            print(encrypted_x_test[row,col].ciphertext(), '|')
    """
    print('=========== Prediction part #1 ===========')
    start = time.time()

    pre_prediction_results = pre_prediction(encrypted_x_test, weights)

    pre_prediction_time = time.time() - start
    print("pre prediction tine:{0}".format(pre_prediction_time) + "[sec]")

    print('=========== Prediction part #2 (client side)===========')
    strat = time.time()

    results, results_classes = calcualte_sigmoid_and_decrypt_result(private_key, pre_prediction_results)

    client_elapsed_time = time.time() - start
    print("client elapsed time:{0}".format(client_elapsed_time) + "[sec]")
    
    print('Predicted values:\t', results)
    print('Predicted classes:\t', results_classes)
    print('Actual classes:\t', y_test)
    print('Accuracy :\t', calculate_accuracy(y_test, results_classes))

    return weights, X_max, X_min

def prediction_api(fname, weights, X_max, X_min):
    # 画像のリサイズ
    api_resize("api_image", fname)  # ./api_image に予測したい画像を入れる
    
    # 画像を 26 次元ベクトルに変換
    in_dim = 26
    img = cv2.imread("./api_resize/" + fname)
    X = np.empty((0, in_dim), int)
    img = np.array(img).reshape(10000, 3).sum(axis=1) / 3
    img = img.reshape(10000, )
    tmp_vec = np.zeros(26)
    for i in range(img.shape[0]):
        tmp_vec[int(img[i]/10)] = tmp_vec[int(img[i]/10)] + 1
    print("tmp_vec")
    print(tmp_vec)
    print(X_max)
    X = np.append(X, [tmp_vec], axis=0)
    X = (X - [X_min]) / ([X_max] - [X_min])
    
    print(X)

    # 鍵ペア取得
    public_key, private_key = pl.generate_paillier_keypair(n_length=256)
    encrypted_res = encrypt_logistic_regression_dataset(public_key, X)
    pre_prediction_res = pre_prediction(encrypted_res, weights)
    results, results_classes = calcualte_sigmoid_and_decrypt_result(private_key, pre_prediction_res)
    
    print('Predicted values:\t', results)
    print('Predicted classes:\t', results_classes)

if __name__ == '__main__':
    
    start = time.time()
    weights, X_max, X_min = logistic_regression()
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

    # unfinished
    fname = "original.jpg"
    prediction_api(fname, weights, X_max, X_min)