from __future__ import division
import idx2numpy
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
    error_rate = ((number_of_labels-number_of_rights)*100/number_of_labels)

    print "Number of images: %d" % number_of_labels
    print "Number of right predictions: %d" % number_of_rights
    print "Error Rate: %" + "%f" % error_rate
    return error_rate

def check_overfitting(rate1, rate2):
    print "Error Rate with 2-fold cross validation: %" + "%f" % ((rate1+rate2)/2)


def reduce_image_size(image_set):
    return numpy.array([a[4:-4, 4:-4] for a in image_set])