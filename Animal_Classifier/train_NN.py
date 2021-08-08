import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os


#Output Paths

output_model = 'output/NN.model'
output_labels= 'output/NN_lb.pickle'
output_plot = 'output/NN_plot.png'

# initialize the data and labels
print("Loading animal images (Cats, Dogs and Pandas)...")
data = []
labels = []

#get the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images('Dataset')))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize the image to be 32x32 pixels
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data,
	labels, test_size=0.25, random_state=42)


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# define the 3072-1024-512-3 architecture using Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 80

# compile the model using SGD as our optimizer and categorical
print("Training the model...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Training
H = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),
	epochs=EPOCHS, batch_size=32)

# Predictions
print("Testing the Model...")
predictions = model.predict(x=X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple Neural Network)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(output_plot)

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(output_model, save_format="h5")
f = open(output_labels, "wb")
f.write(pickle.dumps(lb))
f.close()