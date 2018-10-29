import numpy as np
import pickle


def load_data(datapath, filename):
	data = np.loadtxt(open(datapath + "/" + filename, "rb"), delimiter=",", skiprows=1)		# user, movie, rating, timestamp 
	movieMap = {}
	userMap = {}
	res = np.zeros([len(np.unique(data[:, 0])), len(np.unique(data[:, 1]))]) - 1
	for i, (user, movie, rating, _) in enumerate(data):
		user, movie = int(user), int(movie)
		if movie not in movieMap: movieMap[movie] = len(movieMap)
		if user not in userMap: userMap[user] = len(userMap) 
		res[userMap[user]][movieMap[movie]] = rating
		data[i, 0] = userMap[user]
		data[i, 1] = movieMap[movie]
	pickle.dump(movieMap, open(datapath + "/" + "movieMap", "wb"))
	pickle.dump(userMap, open(datapath + "/" + "userMap", "wb"))
	pickle.dump(res, open(datapath + "/" + "mat", "wb"))
	pickle.dump(data, open(datapath + "/" + "movie_ratings_nudged", "wb"))
	
def calc_loss(true, pred):
    return (true - pred) ** 2

if __name__ == "__main__":
	load_data("/media/zzhuang/00091EA2000FB1D0/iGit/git_projects/RecommendML/data", "movie_ratings.csv")
