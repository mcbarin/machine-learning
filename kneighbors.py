from sklearn.neighbors import KNeighborsClassifier
import time
from methods import idxtoarray, feed_machine, predict_labels, check_results, reduce_image_size


# Train files  idx->numpy array
images = idxtoarray('train-images')
labels = idxtoarray('train-labels')


# Test files idx->numpy array
test_images = idxtoarray('test-images')
test_labels = idxtoarray('test-labels')

# Turn images into 20x20.
# Comment the lines below if you want to run with 28x28 images.
images = reduce_image_size(images)
test_images = reduce_image_size(test_images)

model = KNeighborsClassifier(n_neighbors=5)


start_train = time.time()
feed_machine(images, labels, model)  # Feed the model
end_train = time.time()
print "Training execution time:", (end_train-start_train), "seconds"

start_test = time.time()
predicted_results = predict_labels(test_images, model)  # Predict the test images
end_test = time.time()
print "Prediction execution time:", (end_test-start_test), "seconds"

check_results(predicted_results, test_labels)  # Compare predictions with real values.

# RESULTS
"""
n_neighbors = 3, image size = 28x28
python kneighbors.py
Train action is completed.
Training execution time: 71.2463350296  seconds
Predict action is completed.
Prediction execution time: 917.52027297 seconds
Number of images: 10000
Number of right predictions: 9705
Error Rate: %2.95000
"""


"""
n_neighbors = 3, image size = 20x20
python kneighbors.py
Train action is completed.
Training execution time: 34.130147934 seconds
Predict action is completed.
Prediction execution time: 458.105105877 seconds
Number of images: 10000
Number of right predictions: 9719
Error Rate: %2.81000
"""

"""
n_neighbors = 1, image size = 20x20
python kneighbors.py
Train action is completed.
Training execution time: 30.4881908894 seconds
Predict action is completed.
Prediction execution time: 471.416418076 seconds
Number of images: 10000
Number of right predictions: 9680
Error Rate: %3.200000
"""

"""
n_neighbors = 2  image size = 20x20
python kneighbors.py
Train action is completed.
Training execution time: 37.0344810486 seconds
Predict action is completed.
Prediction execution time: 481.160264969 seconds
Number of images: 10000
Number of right predictions: 9646
Error Rate: %3.540000
"""

"""
n_neighbors = 4  image size = 20x20
python kneighbors.py
Train action is completed.
Training execution time: 32.9988360405 seconds
Predict action is completed.
Prediction execution time: 474.876890182 seconds
Number of images: 10000
Number of right predictions: 9690
Error Rate: %3.100000
"""

"""
n_neighbors = 5  image size = 20x20
python kneighbors.py
Train action is completed.
Training execution time: 35.7740879059 seconds
Predict action is completed.
Prediction execution time: 456.99117589 seconds
Number of images: 10000
Number of right predictions: 9701
Error Rate: %2.990000
"""