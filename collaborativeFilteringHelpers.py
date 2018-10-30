import math

import numpy as np
from numpy import genfromtxt
from scipy.stats import pearsonr


def read_data(csv_dir):
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


def compute_items_similarity_matrix(data):

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


def compute_users_similarity_matrix(data):

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


user_similarity_matrix = compute_users_similarity_matrix(read_data("data/movie_ratings.csv"))
