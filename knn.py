from collections import Counter
#from LinearAlgebra import distance
from scipy.spatial import distance
from stats import mean
import math, random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance.euclidean(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


def predict_preferred_language_by_city(k_values, cities):
    """
    TODO
    predicts a preferred programming language for each city using above knn_classify() and 
    counts if predicted language matches the actual language.
    Finally, print number of correct for each k value using this:
    print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))
    """
    for k in k_values:
        num_correct = 0
        for city in cities:
            tmp_cities = cities.copy()
            tmp_cities.remove(city)
            ret = knn_classify(k,tmp_cities,city[0])
            if ret == city[1]:
                num_correct += 1

        print(k,"neighbor[s]:", num_correct, "correct out of", len(cities))

def get_x_y(cities):
    x = []
    y = []
    for city in cities:
        x.append([city[0][0],city[0][1]])
        y.append(city[1])

    return np.array(x),np.array(y)

def predict_preferred_language_by_city_scikit(k_values, cities):
    for k in k_values:
        num_correct = 0
        knn = KNeighborsClassifier(n_neighbors=k) 
        for city in cities:
            tmp_cities = cities.copy()
            tmp_cities.remove(city)
            x,y = get_x_y(tmp_cities)
            knn.fit(x,y)
            val = knn.predict(np.array([[city[0][0],city[0][1]]]))

            if val == city[1]:
                num_correct += 1

        print(k,"neighbor[s]:", num_correct, "correct out of", len(cities))



if __name__ == "__main__":
    k_values = [1, 3, 5, 7]
    from data import cities
    # Import cities from data.py and pass it into predict_preferred_language_by_city(x, y).
    #print("")
    #predict_preferred_language_by_city(k_values, cities)
    predict_preferred_language_by_city_scikit(k_values, cities)