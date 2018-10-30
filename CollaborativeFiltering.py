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
        self.k = 10 # use the most similar k neighbors
    
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
            pass
                    
    def predict(self, userID, itemID) -> float:
        """
        make prediction for one sample
        return: pred_rating
        """
        # find userID who also rated itemID
        other_userIDs = np.where(self.rating_matrix[:, itemID] != 0)[0]

        # find top K similar users
        users_similarity = self.similarity_matrix[[userID] * len(other_userIDs), other_userIDs]
        similar_users_IDs = np.argpartition(users_similarity, -self.k)[-self.k:]
        print(np.sort(users_similarity))
        print(users_similarity[similar_users_IDs]) # TODO
        # predict


        raise NotImplementedError
    
    def evaluate(self, dataset: myDataset) -> float:
        """
        input: test dataset
        output: loss avg on current test set
        """
        loss_sum = 0
        for userInd, movieInd, rating in dataset:
            userInd, movieInd = int(userInd), int(movieInd)
            pred = self.predict()		# TODO
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
    rec = CollaborativeFiltering(10)
    rec.train("data/movie_ratings.csv")
    rec.predict(0, 1)

