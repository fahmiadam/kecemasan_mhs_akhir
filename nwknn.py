from csv import reader
from sys import exit
from math import sqrt
from operator import itemgetter
import pandas as pd
import numpy as np

class NWKNN:
    def __init__(self, n_neighbor, exp):
        self.n_neighbor = n_neighbor
        self.exp = exp
        pass

    def _find_neighbors(self, distances, k):  #distances= fungsi untuk mengambil data training sebanyak nilai k yang di inputkan
        return distances[0:k]

    def _get_classes(self, training_set): #mengambil atau memuat kelas dari dataset training excel
        return list(set([c[-1] for c in training_set]))

    def predict(self, test):
        neighbor = self.new_neighbor_set
        weight_result = self._weight(self.classes_count, self.exp)
        predictionResult = self._scoring(weight_result, neighbor)
        return predictionResult

    def fit(self, train, test):
        training_set = train
        test_set = test
        distances = []
        dist = 0
        limit = len(training_set[0]) - 1
        
        # Array jumlah data dari setiap kelas
        classes_count = {
            "Tinggi":0,
            "Rendah":0,
            "Sedang":0
        }
        neighbors_set = []
        # generate response classes from training data
        # menghasilkan kelas respons dari data latih
        classes = self._get_classes(training_set)

        try:
            for test_instance in test_set:
                for row in training_set:
                    for x, y in zip(row[:limit], test_instance): #row limit memanggil data selain label
                        dist += (x-y) * (x-y)
                    distances.append(row + [sqrt(dist)])
                    dist = 0

                distances.sort(key=itemgetter(len(distances[0])-1))  #mengurutkan hasil perhitngan jarak tetangga terdekat
                #print(distances[0])
                
                # find k nearest neighbors
                neighbors = self._find_neighbors(distances, k)
                neighbors_set.append(neighbors)

                distances.clear()
            self.new_neighbor_set = []
            for x in neighbors_set:
                x.sort(key = lambda x: x[23],reverse=True)
                self.new_neighbor_set.append(x)
                 
        except Exception as e:
            print(e)

