import idx2numpy
import numpy
import pdb
from sklearn.naive_bayes import GaussianNB


images = idx2numpy.convert_from_file('train-images')
f_read = open('train-images', 'rb')
images = idx2numpy.convert_from_file(f_read)

labels = idx2numpy.convert_from_file('train-labels')
f_read2 = open('train-labels', 'rb')
labels = idx2numpy.convert_from_file(f_read2)

model = GaussianNB()

images2d = images.reshape(len(images),-1)
model.fit(images2d.tolist(),labels.tolist())
print "Fit is done"


# Now images and labels are sent to the Gaussian Naive Bayes

print "Now predict"

test-images = idx2numpy.convert_from_file('test-images')
t_read = open('test-images', 'rb')
test-images = idx2numpy.convert_from_file(t_read)

test-labels = idx2numpy.convert_from_file('test-labels')
t_read2 = open('test-labels', 'rb')
test-labels = idx2numpy.convert_from_file(t_read2)

# pdb.set_trace()



