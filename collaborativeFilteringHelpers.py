import numpy as np

# user_similarity_matrix = compute_users_similarity_matrix(read_data("data/movie_ratings.csv"))
user_similarity_matrix = np.load("data/user_similarity_matrix.npy")

"""
make prediction for one sample
both IDs' indices starts from 1.
return: pred_rating
"""

# index of our arrays starts from 0 so we need to minus 1.
userID = userID - 1
itemID = itemID - 1

# find userID who also rated itemID
other_userIDs = np.where(rating_matrix[:, itemID] != 0)[0]

# find top K similar users
similar_users_IDs = np.where(similarity_matrix[userID, other_userIDs] > 0)[0]  # this is a n*1 vector

num_neighbors = self.k if len(other_userIDs) > k else len(other_userIDs)
similar_users_IDs = np.argpartition(users_similarity, -num_neighbors)[-num_neighbors:]

# predict
denominator = np.sum(np.absolute(users_similarity[similar_users_IDs]))
if denominator > 0:
    pred_rating = np.dot(users_similarity[similar_users_IDs],
                         self.rating_matrix[similar_users_IDs, itemID]) / denominator
else:
    print("denominator is zero")
    return 0

return pred_rating