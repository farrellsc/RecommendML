from utils import calc_loss
from dataProcessor import myDataset
import numpy as np
from numpy import genfromtxt
from scipy.stats import pearsonr
import os.path


class CollaborativeFiltering(object):

    def __init__(self, k=10):
        self.similarity_matrix = None
        self.rating_matrix = None
        self.k = k # use the most similar k neighbors
    
    def train(self, csv_file):
        """
        input: csv_file
        return: loss history by epoch as a list
        """
        self.rating_matrix = self.read_data(csv_file)
        # Computing similarity matrix is time-consuming. We load it if we have already computed.
        if os.path.isfile("data/user_similarity_matrix.npy"):
            self.similarity_matrix = np.load("data/user_similarity_matrix.npy")
        else:
            self.similarity_matrix = self.compute_users_similarity_matrix(self.rating_matrix)
            np.save("data/user_similarity_matrix", self.similarity_matrix)
                    
    def predict(self, userID, itemID) -> float:
        """
        make prediction for one sample
        both IDs' indices starts from 1.
        return: pred_rating
        """

        # index of our arrays starts from 0 so we need to minus 1.
        userID = userID - 1
        itemID = itemID - 1

        # find userID who also rated itemID
        other_userIDs = np.where(self.rating_matrix[:, itemID] != 0)[0]




        # find top K similar users
        other_userIDs_v_similarity = [[id, self.similarity_matrix[userID, id]]
                                      for id in other_userIDs if self.similarity_matrix[userID, id] > 0 and id != userID]
        if len(other_userIDs_v_similarity) == 0:
            return 3 # if no data for this prediction, return 3 as a guess

        other_userIDs_v_similarity.sort(key=lambda x:x[1], reverse=True)
        neighbors_count = self.k if len(other_userIDs_v_similarity) > self.k else len(other_userIDs_v_similarity)
        similar_usersIDs_v_similarity = other_userIDs_v_similarity[:neighbors_count]
        similar_usersIDs_v_similarity_tranposed = list(map(list, zip(*similar_usersIDs_v_similarity))) # tranpose similar_usersIDs_v_similarity
        # predict
        denominator = sum(similar_usersIDs_v_similarity_tranposed[1])
        if denominator > 0:
            pred_rating = np.dot(np.array(similar_usersIDs_v_similarity_tranposed[1]), self.rating_matrix[np.array(similar_usersIDs_v_similarity_tranposed[0]), itemID]) / denominator
        else:
            print("denominator is zero")
            return 0

        return pred_rating

    def evaluate(self, dataset: myDataset) -> float:
        """
        input: test dataset
        output: loss avg on current test set
        """
        loss_sum = 0
        for userInd, movieInd, rating in dataset:
            userInd, movieInd = int(userInd), int(movieInd)
            pred = self.predict(userInd, movieInd)		# TODO
            loss_sum += calc_loss(rating, pred)
        return loss_sum / len(dataset)

    def compute_items_similarity_matrix(self, data):

        # data.shape[1] is the count of items
        pearson_similarity_matrix = np.zeros([data.shape[1], data.shape[1]])

        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                similarity = pearsonr(data[:, i], data[:, j])[0]
                if not np.isnan(similarity):
                    pearson_similarity_matrix[i, j] = similarity
                else:
                    pearson_similarity_matrix[i, j] = 0

        return pearson_similarity_matrix

    def compute_users_similarity_matrix(self, data):

        # data.shape[1] is the count of items
        pearson_similarity_matrix = np.zeros([data.shape[0], data.shape[0]])

        for i in range(data.shape[0]):
            for j in range(data.shape[0]):
                similarity = pearsonr(data[i, :], data[j, :])[0]
                if not np.isnan(similarity):
                    pearson_similarity_matrix[i, j] = similarity
                else:
                    pearson_similarity_matrix[i, j] = 0

        return pearson_similarity_matrix

    def read_data(self, csv_dir):
        """

        :param csv_dir: csv_dir = "data/movie_ratings.csv"
        :return: numpy matrix
        """
        my_data = genfromtxt(csv_dir, delimiter=",", skip_header=1)
        users_count = int(max(my_data[:, 0]))
        items_count = int(max(my_data[:, 1]))

        ratings_matrix = np.zeros([users_count, items_count])

        for i in range(my_data.shape[0]):
            # note that we minus 1 on each index
            ratings_matrix[int(my_data[i, 0]) - 1, int(my_data[i, 1]) - 1] = my_data[i, 2]

        return ratings_matrix


if __name__ == "__main__":
    rec = CollaborativeFiltering(k=5)
    rec.train("data/movie_ratings.csv")
    for i in range(1000):
        for j in range(1000):
            # rec.predict(5, 21)
            # print("\n")
            print(i, j, rec.predict(i, j))
            pass