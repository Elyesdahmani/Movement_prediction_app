# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os


# load a single file as a numpy array
def load_file(filepath):
	dataframe = read_csv(filepath, header=None, delim_whitespace=True)
	return dataframe.values

def align(dataset):
    data = []
    for value in dataset:
        data.append(value[0])
    return data 

def chunks(l, n):
    data = []
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        data.append(l[i:i+n])
    # remove last element for constitency
    data.pop()
    return data

# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
	loaded = list()
	for name in filenames:
		data = load_file(prefix + name)
		norm_data = normalise(data)
		dataset = np.array(chunks(align(norm_data), 128))
		loaded.append(dataset)
	# stack group so that features are the 3rd dimension
	loaded = dstack(loaded)
	return loaded

# normalize test input data
def normalise(dataset):
   i = 0
   max_val = np.max(dataset)
   min_val = np.min(dataset)
   while i<len(dataset):
            dataset[i]= 2*(dataset[i]-min_val)/(max_val-min_val)-1
            #print (dataset[0])
            i=i+1
   return (dataset)


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
	filepath = prefix + group + '/Inertial Signals/'
	# load all 9 files as a single array
	filenames = list()
	# total acceleration
	filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
	# body acceleration
	filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
	# body gyroscope
	filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
	# load input data
	X = load_group(filenames, filepath)
	# load class output
	y = load_file(prefix + group + '/y_'+group+'.txt')
	return X, y

# load the dataset, returns train and test X and y elements
def load_test_dataset(prefix=''):
	
	# load all test
	testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
	print(testX.shape, testy.shape)
	# zero-offset class values
	testy = testy - 1
	# one hot encode y
	testy = to_categorical(testy)
	print(testX.shape, testy.shape)
	return testX, testy

# fit and evaluate a model
def evaluate_model(testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Modèle chargé depuis le disque")
    # model.load_weights("my_model.h5")
    # evaluate model
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(testX, testy, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
    

# run an experiment
def run_experiment(repeats=1):
	# load data
	testX, testy = load_test_dataset()
	# repeat experiment
	for r in range(repeats):
		evaluate_model(testX, testy)
 
# run the experiment
run_experiment()