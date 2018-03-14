# importing the libraries
import numpy as np

# Setting Working directory
import os
os.chdir("D:\Online Study\Machine Learning\Machine Learning A-Z Template Folder\Part 8 - Deep Learning\Section 40 - \
Convolutional Neural Networks (CNN)")
os.getcwd()

# Part 1 - Building the CNN
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Starting the timer
from timeit import default_timer as timer
start = timer()

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000/32)

from keras.preprocessing import image

img = image.load_img('./bird4.jpg', target_size = (64, 64))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)
result = classifier.predict(img)

# training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print('dog')
else:
    prediction = 'cat'
    print('cat')

# elapsed time in seconds
end = timer()
print(end - start)
print('Your program is finished')
