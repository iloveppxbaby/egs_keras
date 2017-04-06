from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# 1. Load Data and split into input (X) and output (Y) variables

dataset = np.ones((1000, 9))
X = dataset[:, 0:8]
Y = dataset[:, 8]

print X.shape
print Y.shape
print np.hstack((X, Y[:, np.newaxis]))

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
