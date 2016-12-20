from sklearn.naive_bayes import GaussianNB
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

model = GaussianNB()  # Gaussian Naive Bayes Classifier


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
Image file size: 28x28

python naive_bayes.py
Train action is completed.
Training execution time: 7.80242204666 seconds
Predict action is completed.
Prediction execution time: 0.949084043503 seconds
Number of images: 10000
Number of right predictions: 5558
Error Rate: %44.420000
"""


"""
When image file turned into 20x20

python naive_bayes.py
Train action is completed.
Training execution time: 3.82516789436 seconds
Predict action is completed.
Prediction execution time: 0.49881696701 seconds
Number of images: 10000
Number of right predictions: 7470
Error Rate: %25.300000
"""