import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

points = {'blue': [[2,4], [1,3], [2,3], [3,2], [2,1]],
          'orange': [[5,6], [4,5], [4,6], [6,6], [5,4]]}

new_point = [3,3]

def euclidian_distance(p,q):
    return np.sqrt(np.sum((np.array(p)-np.array(q))**2))


class KNN:
    def __init__(self, k):
        self.k = k
        self.points = None
    
    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distnaces = []
        for group in self.points:
            for point in self.points[group]:
                distnaces.append((euclidian_distance(point, new_point), group))
        distnaces.sort()
        k_nearest = distnaces[:self.k]
        k_nearest = [i[1] for i in k_nearest]
        return Counter(k_nearest).most_common(1)[0][0]


knn = KNN(3)
knn.fit(points)
k_nearest = knn.predict(new_point)
print(k_nearest)