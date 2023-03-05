
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv) # image processing
from PIL import Image  # image processing
# import lodgepole.image_tools as lit # linear approximation of gamma correction KAGGLE WILL NOT IMPORT THIS
import os
from skimage import color
from skimage import io
import random
from sklearn.decomposition import PCA
from scipy import spatial
import math

people = []  # list of randomly selected people of which there are least 50 pictures
while len(people) < 10:
    r = random.randint(1, 4000)
    _, _, files = next(os.walk('/kaggle/input/11785-spring2021-hw2p2s1-face-classification/train_data/' + str(r)))
    file_count = len(files)
    # list = os.listdir('/kaggle/input/11785-spring2021-hw2p2s1-face-classification/train_data/'+str(x)) # dir is
    # your directory path file_count = len(list)
    if file_count >= 50:
        people.append(r)

# color_img = np.asarray(Image.open(img_filename)) / 255
# gray_img = lit.rgb2gray_approx(color_img) 
# KAGGLE WONT IMPORT lit, skimage used instead below
training = list()
# dataset = [[],[],[],[],[],[],[],[],[],[]] #contains all the images to be used, dataset[0]=person1,
# dataset[1]=person2,..., dataset[0][0]= person1's first photo, ... i=0 #i-th person
for p in people:
    c = 0
    for dirname, _, filenames in os.walk(
            '/kaggle/input/11785-spring2021-hw2p2s1-face-classification/train_data/' + str(p)):
        for filename in filenames:
            if c < 50:  # TODO loop executes doing nothing if c>50
                img = color.rgb2gray(io.imread(
                    '/kaggle/input/11785-spring2021-hw2p2s1-face-classification/train_data/' + str(p) + '/' + filename))
                training.append(
                    img.flatten())  # dataset[i].append(img.flatten()) #images need to be flat (vectors and not arrays)
                c += 1
    # i+=1



principal_components = list()

for j in [25, 50, 100]:
    pca = PCA(n_components=j)
    principal_components.append(pca.fit_transform(training))

# Found array with dim 3. Estimator expected <= 2.
# ValueError: n_components=100 must be between 0 and min(n_samples, n_features)=50 with svd_solver='full'
# Expected 2D array, got 1D array instead


K = 10  # number of classes
max_iterations = 100


def euclidean_dist(value1, value2):
    result = 0
    for i in range(len(value1)):
        result = result + ((value1[i] - value2[i]) ** 2)
    return math.sqrt(result)


def cosine_dist(value1, value2):
    return spatial.distance.cosine(value1, value2)


def purity(clusters):
    return sum(clusters) / len(principal_components[0])


def K_means(M, function):
    # M value is 0 for 25 dimension, 1 for 50 and 2 for 100
    X = principal_components[M]
    clusters = [[] for _ in range(K)]  # we need as many clusters as the classes
    centroids = []

    # first we need to initialize the centroids randomly
    for j in range(K):
        index = random.randint(0, len(X))
        centroids.append(X[index])

    # For each iteration
    iteration = 0
    while iteration <= max_iterations:

        # Creating clusters
        # for each x value we need to compute the distance with each centroid
        for i in range(len(X)):
            each_distance = []
            for j in range(K):
                if function == "euclidean":
                    each_distance.append(euclidean_dist(X[i], centroids[j]))
                else:  # cosine distance
                    each_distance.append(cosine_dist(X[i], centroids[j]))
            q = np.argmin(each_distance)
            clusters[q].append(i)

        # Now we have our clusters ready and we need to compute the new centroids
        prev_centroids = centroids
        for j in range(K):
            x_sum = 0
            for i in X:
                if i in clusters[j]:
                    x_sum += X[i]
            centroids[j] = 1 / len(clusters[j]) * x_sum
            # We check if the centroids are not altered
            if function == "euclidean":
                differences.append(euclidean_dist(prev_centroids[j], centroids[j]))
            else:
                differences = cosine_dist(prev_centroids[j], centroids[j])

        if sum(differences) == 0:
            break

        iteration += 1
        print(centroids[0])
