
import numpy
numpy.random.seed(1)
import pandas
import sys
import os
from keras.models import Sequential, Model
from keras.layers import Dense, Input,concatenate, LSTM, Activation, Dropout,Flatten
from keras.optimizers import Adagrad, Adam,SGD,RMSprop
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomUniform,Ones,RandomNormal
from keras import regularizers
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing, decomposition, metrics
from sklearn.utils import class_weight,resample
#from imblearn.over_sampling import ADASYN

from file_handler import savetxt

def sampling_SMOTE(x,y):
	sm = ADASYN()
	x_res, y_res = sm.fit_sample(x,y)
	return x_res,y_res 

def readDataset(inf,isResample,isreplace):
	# load dataset
	dataframe = pandas.read_csv(inf)
	if (isResample==1):
		dataframe = resample(dataframe,replace=isreplace)
	dataset = dataframe.values
	feature_num = len(dataset[0,:])-1
	nid = dataset[:,0]
	x = dataset[:,1:feature_num].astype(float)
	y = dataset[:,feature_num]
	return x, y, nid

def datasetPreprocess(x_train,x_test):
	mmscaler = preprocessing.MinMaxScaler(feature_range=(-10,10))
	mmscaler.fit(x_train) 
	#x_train = mmscaler.transform(x_train)
	#x_test = mmscaler.transform(x_test) 

	#x_train = preprocessing.no(x_train)
	#x_test  = preprocessing.scale(x_test)

	stdscaler = preprocessing.StandardScaler()
	stdscaler.fit(x_train) 
	x_train = stdscaler.transform(x_train)
	x_test = stdscaler.transform(x_test) 
	
	#x_train = preprocessing.scale(x_train)
	#x_test  = preprocessing.scale(x_test )

	return x_train,x_test

def classLabelEncoder(y_train,y_test):
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_ytr = encoder.transform(y_train)
	encoded_yte = encoder.transform(y_test)

	# convert integers to dummy variables (i.e. one hot encoded)
	dummy_ytr = np_utils.to_categorical(encoded_ytr)
	dummy_yte = np_utils.to_categorical(encoded_yte)
	return dummy_ytr,dummy_yte

def createModel_MLP(feature,cls):
	feature_num = feature.shape[1]
	cls_num = cls.shape[1]
	model = Sequential()

	model.add(Dense(output_dim=feature_num,input_dim=feature_num))#, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	model.add(Dense(output_dim=feature_num))#,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	#--Output Layer
	model.add(Dense(output_dim=cls_num))
	model.add(Activation('softmax'))
	
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
	return model

def createModel_MLP_sentence(feature,cls):
	feature_num = feature.shape[1]
	cls_num = cls.shape[1]
	model = Sequential()

	model.add(Dense(units=feature_num,input_dim=feature_num))#, kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(units=64))#,kernel_regularizer=regularizers.l2(0.01),bias_regularizer=regularizers.l1(0.01)))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	
	#--Output Layer
	model.add(Dense(units=2))
	model.add(Activation('softmax'))
	
	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
	return model

def getTrueClass(y):
	y_true = []
	for i in y:
		y = i.tolist()
		idxClass = y.index(max(y))
		y_true.append(idxClass)
	return y_true

def getClassList(dummy_y,y_true):
	y_value = []
	y_label = []
	y_idx = []
	y_value = getTrueClass(dummy_y)
	
	for i in range (0,len(y_value)):
		if not(y_value[i] in y_idx):
			y_idx.append(y_value[i])
			y_label.append(y_true[i])
	return y_label,y_idx

def convert_binary_class(y):
	for i in range (0,len(y)):
		if 'class2' in y[i]:
			y[i]='class1'
	return y
def nominalize_class(y):
	for i in range (0,len(y)):
		if 'class' not in str(y[i]):
			y[i]='class'+str(y[i])
	return y


def datasetPreprocess(x_train,x_test):
  mmscaler = preprocessing.MinMaxScaler(feature_range=(-1,1)) 
  mmscaler.fit(x_train)
  x_train = mmscaler.transform(x_train)
  x_test = mmscaler.transform(x_test)
  ''' 
  stdscaler = preprocessing.RobustScaler()
  stdscaler.fit(x_train) 
  x_train = stdscaler.transform(x_train)
  x_test = stdscaler.transform(x_test)
  '''
  return x_train,x_test

#--------------------MAIN PROGRAM-------------------------------------
train_file  = sys.argv[1]#"feature_compare/train_FT1-SAM0n-BC-PTrue.csv"
test_file  = sys.argv[2]#"feature_compare/train_FT1-SAM0n-BC-PTrue.csv"
outf   = sys.argv[3]#"TR-DNN_FT1-SAM0n-BC-PTrue.txt
out_model = sys.argv[4]
resultStr  = "\n===============DNN RESULT==============\n"
resultStr  += "\nTrain Data\t"+train_file
resultStr  += "\nTest Data\t"+test_file

corr=0
resSpk=[]
resStat =[]
resNumSpk = []

if not(os.path.exists(train_file)) or not(os.path.exists(test_file)):
	print("Input file error")
	sys.exit()

#Define data set
x_train,y_train, nid_train = readDataset(train_file,1,True)
x_test,y_test, nid_test = readDataset(test_file,0,False)
#y_train = nominalize_class(y_train)
#y_test = nominalize_class(y_test)
#convert to binary class
print(list(set(y_train)))
print(list(set(y_test)))
import math
yt = y_train.tolist()
for i in range(0,len(yt)):
		if yt[i]!=yt[i]:
			 print(str(i)+str(nid_train[i+1]))
#y_train = convert_binary_class(y_train)
#y_test = convert_binary_class(y_test)

#x_train,y_train = sampling_SMOTE(x_train,y_train)
#x_train,x_test = datasetPreprocess(x_train,x_test)
y_train_enc,y_test_enc = classLabelEncoder(y_train,y_test)
y_label,y_index = getClassList(y_train_enc,y_train)

resultStr += "\n\nFeature number\t"+str(x_train.shape[1])
resultStr += "\nClass number\t"+str(len(list(set(y_train))))
resultStr += "\nTrain sample\t"+str(len(x_train))
resultStr += "\nTest sample\t\t"+str(len(x_test))

#BuildModel

model = createModel_MLP(x_train,y_train_enc)
cls_weight = class_weight.compute_class_weight(class_weight='balanced',classes=numpy.unique(y_train),y=y_train)

epoNum = 300

model.fit(x_train, y_train_enc, epochs=epoNum, validation_split=0.1, verbose=1,class_weight=cls_weight)

modelEval = model.evaluate(x_test,y_test_enc)
predict = model.predict(x_test) 
resultStr += "\n"+str(model.get_config())+"\n\n"

resultStr += "\nEpoch number\t"+ str(epoNum)
resultStr += "\n\nEvaluation acc\t"+str(modelEval[1])
resultStr += "\nEvaluation loss\t"+str(modelEval[0])+"\n"
resultStr += str(y_label,)+str(y_index)+"\n\n"

resultStr2 = "ID-word\tactual\tpredicted\tarray\n"
prediction_label = []

y_label = sorted(list(set(y_train)))

for term in range(0,len(predict)):

	p = predict[term].tolist()
	predicted = p.index(max(p))
	y = y_test_enc[term].tolist()
	ground = y.index(max(y))

	resultStr2 += str(nid_test[term])+"\t"+str(y_test[term])+"\t"+str(y_label[predicted])+"\t"+str(p)+"\t"+str(y)+"\n"
	#prediction_label.append('class'+str(predicted))
	prediction_label.append(y_label[predicted])



resultStr += metrics.classification_report(y_test.tolist(),prediction_label,digits=6)+"\n\n"+resultStr2
savetxt(resultStr,outf)
model.save(out_model)
