from sklearn.neighbors import KNeighborsClassifier
import time
from methods import idxtoarray, feed_machine, predict_labels, check_results, reduce_image_size, check_overfitting


# Train files  idx->numpy array
images_1 = idxtoarray('train-images')
labels_1 = idxtoarray('train-labels')


# Test files idx->numpy array
images_2 = idxtoarray('test-images')
labels_2= idxtoarray('test-labels')

# Turn images into 20x20.
# Comment the lines below if you want to run with 28x28 images.
images_1 = reduce_image_size(images_1)
images_2 = reduce_image_size(images_2)

model = KNeighborsClassifier(n_neighbors=3)


start_train = time.time()
feed_machine(images_1, labels_1, model)  # Feed the model for training set 1
end_train = time.time()
print "Training #1 execution time:", (end_train-start_train), "seconds"

start_test = time.time()
predicted_results1 = predict_labels(images_2, model)  # Predict the test images  for training set 1
end_test = time.time()
print "Prediction #1 execution time:", (end_test-start_test), "seconds"

rate_1 = check_results(predicted_results1, labels_2)  # Compare predictions with real values.

start_train = time.time()
feed_machine(images_2, labels_2, model)  # Feed the model for training set 2
end_train = time.time()
print "Training #2 execution time:", (end_train-start_train), "seconds"

start_test = time.time()
predicted_results2 = predict_labels(images_1, model)  # Predict the test images  for training set 2
end_test = time.time()
print "Prediction #2 execution time:", (end_test-start_test), "seconds"

rate_2 = check_results(predicted_results2, labels_1)  # Compare predictions with real values.

check_overfitting(rate_1,rate_2) #take the average of two outcomes

# RESULTS
#for 20x20 images
"""
    Train action is completed.
    Training #1 execution time: 31.4408240318 seconds
    Predict action is completed.
    Prediction #1 execution time: 369.799671888 seconds
    Number of images: 10000
    Number of right predictions: 9719
    Error Rate: %2.810000
    Train action is completed.
    Training #2 execution time: 1.11668491364 seconds
    Predict action is completed.
    Prediction #2 execution time: 409.224318981 seconds
    Number of images: 60000
    Number of right predictions: 56538
    Error Rate: %5.770000
    Error Rate with 2-fold cross validation: %4.290000
"""
#for 28x28 images
"""
    Train action is completed.
    Training #1 execution time: 42.8424470425 seconds
    Predict action is completed.
    Prediction #1 execution time: 667.538208008 seconds
    Number of images: 10000
    Number of right predictions: 9705
    Error Rate: %2.950000
    Train action is completed.
    Training #2 execution time: 1.98010802269 seconds
    Predict action is completed.
    Prediction #2 execution time: 758.222872972 seconds
    Number of images: 60000
    Number of right predictions: 56571
    Error Rate: %5.715000
    Error Rate with 2-fold cross validation: %4.332500
"""