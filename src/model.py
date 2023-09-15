import numpy as np
from collections import defaultdict
import inspect
import sys
import matplotlib.pyplot as plt
import os




class RBFNTrain:
    def __init__(self, data_path, k, epsilon, lr, epoch, gamma,pseudo_inverse=False):
        self.data = np.array(np.loadtxt(data_path[0]))
        self.k = k
        self.epsilon = epsilon
        self.lr = lr
        self.epoch = epoch
        self.gamma = gamma
        self.pseudo_inverse = pseudo_inverse


    def euclidean_dist(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def k_means(self,data_input):
        # init cluster_center_array
        cluster_center_array = []
        for i in range(self.k):
            cluster_center_array.append(np.array(data_input[np.random.randint(data_input.shape[0])]))
        cluster_center_array = np.array(cluster_center_array)
        while 1:
            current_cluster_center_array = cluster_center_array.copy()
            update_candidates = defaultdict(list)
            print("cluster cen array")
            print(cluster_center_array)
            for data_index, data_element in enumerate(data_input):
                dist = [self.euclidean_dist(data_element, center) for center in cluster_center_array]
                dist = np.array(dist)
                # print(dist)
                current_center_index = np.argmin(dist)
                # print(current_center_index)
                update_candidates[current_center_index].append(np.array(data_element))
                # drop first column of zeros
                # update_candidates = update_candidates[:,1:]
            for i in range(self.k):
                if (len(update_candidates[i])) == 0:
                    continue
                update_val = np.vstack(update_candidates[i])
                update_val = np.array(update_val)
                # print(update_val)
                update_val = update_val.mean(axis=0)
                # print(update_val)
                cluster_center_array[i] = update_val

            if self.euclidean_dist(current_cluster_center_array, cluster_center_array) < self.epsilon:
                break


        return cluster_center_array, update_candidates

    def rbfn_init(self, centers, update_candidates):
        rbfn_array = dict.fromkeys(range(len(centers)))
        for index, center in enumerate(centers):
            def rbfn(x, center=center, gamma=self.std(np.array(update_candidates[index]))):
                #2 * gamma ** 2
                return np.exp(-1 * self.euclidean_dist(x, center) ** 2 / (2 * gamma ** 2))

            rbfn_array[index] = rbfn
        return rbfn_array


    def rbfn_pseudo_init(self,centers, gamma):
        rbfn_array = dict.fromkeys(range(len(centers)))

        for index, center in enumerate(centers):
            def rbfn(x, center=center, gamma=gamma):
                return np.exp(-1 * self.euclidean_dist(x, center) ** 2 *gamma)

            rbfn_array[index] = rbfn
        return rbfn_array


    def std(self,data):
        return np.ndarray.std(data)


    def weights_init(self,k):
        return np.random.randn(k)


    def gaussian_transform(self,x, rbfns_dict):
        return np.array([rbfns_dict[i](x) for i in range(len(rbfns_dict))])


    def train_rbfn(self):
        data_input = np.delete(self.data, -1, axis=1)
        input_dim = data_input.shape[1]
        data_output = self.data[:, -1]
        # init
        weights = self.weights_init(self.k)
        bias = np.random.rand(1)

        if self.pseudo_inverse:
            rbfn_centers_pseudo = data_input
            #100 or 50 for 4d with /(2 * gamma ** 2)
            #0.0002 or 0.00005 for 4d with gamma
            #0.000001 for 4d with gamma in exe
            #0.00001 for 6d with gamma
            rbfns_dict_pseudo = self.rbfn_pseudo_init(rbfn_centers_pseudo,self.gamma)

            psudo_matrix = []

            for i in range(len(rbfns_dict_pseudo)):
                rbfn_v = np.vectorize(rbfns_dict_pseudo[i], signature="(m)->()")
                column = np.array([rbfn_v(data_input)])
                column = column.T
                psudo_matrix.append(column)
            psudo_matrix = np.hstack(psudo_matrix)
            print(psudo_matrix)
            print(psudo_matrix.shape)
            data_output = data_output - bias
            weights = np.linalg.inv(
                psudo_matrix.T @ psudo_matrix + 1e-9* np.eye(psudo_matrix.shape[0])) @ psudo_matrix.T @ data_output
            forward = []
            for self.data in data_input:
                forward.append(np.dot(weights, self.gaussian_transform(self.data, rbfns_dict_pseudo)) + bias)
            forward = np.array(forward).reshape(len(data_input))
            loss_history = []
            loss = np.mean((1 / 2) * ((data_output - forward) ** 2))
            loss_history.append(loss)
            loss_history = np.array(loss_history)
            return weights, bias, rbfns_dict_pseudo, loss_history,input_dim
        else:
            rbfn_centers, update_candidates = self.k_means(data_input)
            rbfns_dict = self.rbfn_init(rbfn_centers, update_candidates)
            for i in range(self.epoch):
                forward = []
                for data in data_input:
                    forward.append(np.dot(weights, self.gaussian_transform(data, rbfns_dict)) + bias)
                forward = np.array(forward).reshape(len(data_input))
                print("forward")
                print(forward)
                print("data_output")
                print(data_output)
                loss_history = []
                loss = np.mean((1 / 2) * ((data_output - forward) ** 2))
                loss_history.append(loss)

                for i in range(len(weights)):
                    rbfn_v = np.vectorize(rbfns_dict[i], signature="(m)->()")
                    weights[i] += self.lr * np.mean((data_output - forward) * (rbfn_v(data_input)))
                bias += self.lr * (np.mean(data_output - forward))
                loss_history = np.array(loss_history)
            return weights, bias, rbfns_dict, loss_history,input_dim

class RBFN(RBFNTrain):
    def __init__(self, weights, bias, rbfns_dict):
        self.weights = weights
        self.bias = bias
        self.rbfns_dict = rbfns_dict

    def predict(self, x):
            return np.dot(self.weights, super().gaussian_transform(x, self.rbfns_dict)) + self.bias








# weights, bias, rbfns_dict, loss_history = RBFNTrain(data, 5, 0.000001, 0.5, 100, pseudo_inverse=True).train_rbfn()
# print(weights)
# print(loss_history)
# print(loss_history.shape)
# rbfn = RBFN(weights, bias, rbfns_dict)
# print(rbfn.predict(np.array([18.4084, 11.4408, 09.0715])))