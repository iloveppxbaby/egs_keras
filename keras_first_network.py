from keras.models import Sequential
from keras.layers import Dense
import numpy

# 1. Load Data
#    1.1 fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#    1.2 Download the dataset Pima Indians onset of diabetes dataset, which is a standard machine learining dataset from the UCI Machine learning repository
#        wget http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data

dataset = numpy.loadtxt("pima-indians-diabetes.data", delimiter=",")
#    1.3 split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]

# 2. Define Model

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# 3. Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. Fit Model
model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

# 5. Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
