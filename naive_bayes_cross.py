from sklearn.naive_bayes import GaussianNB
import time
from methods import idxtoarray, feed_machine, predict_labels, check_results, reduce_image_size, check_overfitting


# Train1 files  idx->numpy array
images_1 = idxtoarray('train-images')
labels_1 = idxtoarray('train-labels')


# Train2 files idx->numpy array
images_2 = idxtoarray('test-images')
labels_2 = idxtoarray('test-labels')

# Turn images into 20x20.
# Comment the lines below if you want to run with 28x28 images.
images_1 = reduce_image_size(images_1)
images_2 = reduce_image_size(images_2)

model = GaussianNB()  # Gaussian Naive Bayes Classifier

start_train = time.time()
feed_machine(images_1, labels_1, model)  # Feed the model for training set 1
end_train = time.time()
print "Training #1 execution time:", (end_train-start_train), "seconds"

start_test = time.time()
predicted_results1 = predict_labels(images_2, model)  # Predict the test images for training set 1
end_test = time.time()
print "Prediction #1 execution time:", (end_test-start_test), "seconds"

rate_1 = check_results(predicted_results1, labels_2)  # Compare predictions with real values.

start_train = time.time()
feed_machine(images_2, labels_2, model)  # Feed the model for training set 2
end_train = time.time()
print "Training #2 execution time:", (end_train-start_train), "seconds"

start_test = time.time()
predicted_results2 = predict_labels(images_1, model)  # Predict the test images for training set 2
end_test = time.time()
print "Prediction #2 execution time:", (end_test-start_test), "seconds"

rate_2 = check_results(predicted_results2, labels_1)  # Compare predictions with real values.

check_overfitting(rate_1,rate_2) #take the average of two outcomes

# RESULTS

#for 20x20 images
"""
    Train action is completed.
    Training #1 execution time: 4.57290697098 seconds
    Predict action is completed.
    Prediction #1 execution time: 0.507277011871 seconds
    Number of images: 10000
    Number of right predictions: 7470
    Error Rate: %25.300000
    Train action is completed.
    Training #2 execution time: 0.629858016968 seconds
    Predict action is completed.
    Prediction #2 execution time: 4.57053089142 seconds
    Number of images: 60000
    Number of right predictions: 42154
    Error Rate: %29.743333
    Error Rate with 2-fold cross validation: %27.521667
"""
#for 28x28 images
"""
    Train action is completed.
    Training #1 execution time: 8.88528490067 seconds
    Predict action is completed.
    Prediction #1 execution time: 1.01989197731 seconds
    Number of images: 10000
    Number of right predictions: 5558
    Error Rate: %44.420000
    Train action is completed.
    Training #2 execution time: 1.22138094902 seconds
    Predict action is completed.
    Prediction #2 execution time: 8.25105786324 seconds
    Number of images: 60000
    Number of right predictions: 34071
    Error Rate: %43.215000
    Error Rate with 2-fold cross validation: %43.817500
"""