# import the necessary libraries
from tensorflow.keras.models import load_model
import pickle
import cv2


#PATHS TO OUTPUTS FROM TRAINED MODEL!
Input_Image = 'test_images/cute_dog.jpg' #input image path\name!!!!
Model_Path = 'output/NN.model'
label_path = 'output/NN_lb.pickle'
width = 32 # 28 default
height = 32 # 28 default

# load the input image and resize it spatial dimensions
image = cv2.imread(Input_Image)
output = image.copy()
image = cv2.resize(image, (width, height))

# scale the pixel values between [0, 1]
image = image.astype("float") / 255.0


image = image.flatten()
image = image.reshape((1, image.shape[0]))

# loading the model and label binarizer
print("Loading the network and Label binarizer for Animal Classification...")
model = load_model(Model_Path)
lb = pickle.loads(open(label_path, "rb").read())

# make a prediction on the image
predictions = model.predict(image)

# find the class label index with the largest corresponding probability
i = predictions.argmax(axis=1)[0]
label = lb.classes_[i]

# draw the class label and probability on the output image
text = "{}: {:.2f}%".format(label, predictions[0][i] * 100)
cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
	(0, 255, 255), 2)

# show the output image
cv2.imshow("Image", output)
cv2.waitKey(0)