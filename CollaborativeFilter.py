import numpy as np
from numpy import genfromtxt

from utils import calc_loss
from dataProcessor import myDataset


class CollaborativeFilter:
    def __init__(self, dim, userNum, movieNum, epoch, lr):
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.userNum = userNum
        self.movieNum = movieNum
        #---------------------------------define weights here---------------------------------
    
    def train(self, dataset: myDataset) -> list:
        """
        input: train dataset
        return: loss history by epoch as a list
        """

        return None
                    
    def predict(self) -> float:
        """
        make prediction for one sample
        return: pred_rating
        """
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

    def read_data(csv_dir):
        """

        :param csv_dir: csv_dir = "data/movie_ratings.csv"
        :return: numpy matrix
        """
        my_data = genfromtxt(csv_dir, delimiter=",", skip_header=1)
        users_count = int(max(my_data[:, 0]))
        items_count = int(max(my_data[:, 1]))

        ratings_matrix = np.zeros([users_count, items_count])

        # note that we minus 1 on each index
        ratings_matrix[int(my_data[i, 0]) - 1, int(my_data[i, 1]) - 1] = my_data[i, 2]

        return ratings_matrix




if __name__ == "__main__":
    pass

