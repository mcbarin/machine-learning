import idx2numpy
import numpy
import pdb
from sklearn.naive_bayes import GaussianNB


# Train files  idx->numpy array
images = idx2numpy.convert_from_file('train-images')
f_read = open('train-images', 'rb')
images = idx2numpy.convert_from_file(f_read)

labels = idx2numpy.convert_from_file('train-labels')
f_read2 = open('train-labels', 'rb')
labels = idx2numpy.convert_from_file(f_read2)


# Test files idx->numpy array
test_images = idx2numpy.convert_from_file('test-images')
t_read = open('test-images', 'rb')
test_images = idx2numpy.convert_from_file(t_read)

test_labels = idx2numpy.convert_from_file('test-labels')
t_read2 = open('test-labels', 'rb')
test_labels = idx2numpy.convert_from_file(t_read2)

model = GaussianNB()

images2d = images.reshape(len(images), -1)
model.fit(images2d.tolist(), labels.tolist())
print "Fit is done"

test_images2d = test_images.reshape(len(test_images), -1)
print "Predict is done"

predicted = model.predict(test_images2d)
print predicted

# pdb.set_trace()


predicted_list = predicted.tolist()
test_labels_list = test_labels.tolist()

number_of_rights = 0
number_of_labels = len(test_labels_list)

for index in range(0, number_of_labels):
    if predicted_list[index] == test_labels_list[index]:
        number_of_rights += 1

print "Number of images: %d" % number_of_labels
print "Number of rights: %d" % number_of_rights

