import tensorflow as tf
from tensorflow import keras

# This is a super simple image classifier
# It basically only trains on a some images
# and does not do any validation\n
# Load data for the model
( train_images, train_labels), ( test_images, test_labels) = keras.datasets.cifar10.load_data()   

# Normalize images to [0,1] range
train_images = train_images.astype('float32') / 255.
# Treat test images the same 

test_images = test_images.astype('float32') / 255.0

# create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    # using small dense layers
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)  #10 for the number of classes
])

# Compiling the model 
# using categorical_crossentropy as that seems to be the common one
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Starting to train the model
# iterating multiple times over the training data 
model.fit(train_images, train_labels, epochs=5) # I might change this later

# evaluating the model
# since I want to see how well it does with test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print(f"Model accuracy: {test_acc}") # Simple print to check accuracy

