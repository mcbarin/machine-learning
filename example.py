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

pdb.set_trace()

model.fit(images,labels)



