import numpy
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)



test_images2d = test_images.reshape(len(test_images),-1)

images2d = images.reshape(len(images),-1)

model.fit(images2d,labels)

model.fit(test_images2d,test_labels)



predicted = model.predict(test_images2d)

#pdb.set_trace()

predicted_list = predicted.tolist()

test_labels_list = test_labels.tolist()



number_of_rights = 0

number_of_labels = len(test_labels_list)


for index in range(0, number_of_labels):
    if predicted_list[index] == test_labels_list[index]:
        number_of_rights += 1

print "Number of images: %d" % number_of_labels
print "Number of rights: %d" % number_of_rights