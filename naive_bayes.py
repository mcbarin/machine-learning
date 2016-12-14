import idx2numpy
from sklearn.naive_bayes import GaussianNB


def idxtoarray(filename):
    numpy_array = idx2numpy.convert_from_file(filename)
    f_read = open(filename, 'rb')
    numpy_array = idx2numpy.convert_from_file(f_read)
    return numpy_array


def feed_machine(images, labels, model):
    image_size = len(images)
    images2d = images.reshape(image_size, -1)
    model.fit(images2d.tolist(), labels.tolist())
    print "Fit action is completed."


def predict_labels(images, model):
    image_size = len(images)
    test_images2d = test_images.reshape(image_size, -1)
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

model = GaussianNB()  # Gaussian Naive Bayes Classifier

feed_machine(images, labels, model)  # Feed the model
predicted_results = predict_labels(test_images, model)  # Predict the test images
check_results(predicted_results, test_labels)  # Compare predictions with real values.

