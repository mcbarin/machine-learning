import idx2numpy
from sklearn.naive_bayes import GaussianNB
import time
import numpy


def idxtoarray(filename):
    f_read = open(filename, 'rb')
    numpy_array = idx2numpy.convert_from_file(f_read)
    return numpy_array


def feed_machine(images, labels, model):
    image_size = len(images)
    images2d = images.reshape(image_size, -1)
    model.fit(images2d.tolist(), labels.tolist())
    print "Train action is completed."


def predict_labels(images, model):
    image_size = len(images)
    test_images2d = images.reshape(image_size, -1)
    predicted_results = model.predict(test_images2d)
    print "Predict action is completed."
    return predicted_results


def check_results(predicted, answers):
    predicted_list = predicted.tolist()
    test_labels_list = answers.tolist()
    number_of_rights = 0
    number_of_labels = len(test_labels_list)
    for index in range(0, number_of_labels):
        if predicted_list[index] == test_labels_list[index]:
            number_of_rights += 1

    print "Number of images: %d" % number_of_labels
    print "Number of right predictions: %d" % number_of_rights

# Train files  idx->numpy array
images = idxtoarray('train-images')
labels = idxtoarray('train-labels')


# Test files idx->numpy array
test_images = idxtoarray('test-images')
test_labels = idxtoarray('test-labels')

# Turn images into 20x20
images = numpy.array([a[4:-4, 4:-4] for a in images])
test_images = numpy.array([a[4:-4, 4:-4] for a in test_images])

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


"""
Image file size: 28x28

python naive_bayes.py
Train action is completed.
Training execution time: 8.29062891006 seconds
Predict action is completed.
Prediction execution time: 0.972679853439 seconds
Number of images: 10000
Number of right predictions: 5558
"""


"""
When image file turned into 20x20

python naive_bayes.py
Train action is completed.
Training execution time: 3.48154807091 seconds
Predict action is completed.
Prediction execution time: 0.496382951736 seconds
Number of images: 10000
Number of right predictions: 7470
"""