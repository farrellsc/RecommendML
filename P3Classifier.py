import numpy as np
import tqdm
from utils import calc_loss
from dataProcessor import myDataset, CVdataloader


class P3Classifier:
    def __init__(self, dim, userNum, movieNum, epoch, lr):
        self.dim = dim
        self.epoch = epoch
        self.lr = lr
        self.userNum = userNum
        self.movieNum = movieNum
        self.userW = np.random.randn(userNum, dim) * 0.01
        self.movieW = np.random.randn(movieNum, dim) * 0.01
        self.userFlag = np.zeros([userNum])
        self.movieFlag = np.zeros([movieNum])
        self.userAvg = None
        self.movieAvg = None
    
    def train(self, dataset: myDataset) -> list:
        """
        input: train dataset
        return: loss history by epoch as a list
        """
        loss_history = []
        for i in range(self.epoch):
            preds = []
            for batch_ind, (userInd, movieInd, rating) in enumerate(dataset):
                userInd, movieInd = int(userInd), int(movieInd)
                self.userFlag[userInd] = 1
                self.movieFlag[movieInd] = 1
                pred = self.predict(userInd, movieInd)
                preds.append(pred)
                userG = 2 * self.lr * (rating - pred) * self.movieW[movieInd, :]
                movieG = 2 * self.lr * (rating - pred) * self.userW[userInd, :]
                self.userW[userInd, :] += userG
                self.movieW[movieInd, :] += movieG
            loss_history.append(calc_loss(dataset.getY().flatten(), np.array(preds)).sum() / len(dataset))
        self.userAvg = np.sum(self.userW, axis=0) / self.userW.shape[0]
        self.movieAvg = np.sum(self.movieW, axis=0) / self.movieW.shape[0]
        return loss_history
                    
    def predict(self, userInd, movieInd) -> float:
        """
        make prediction for one sample
        """
        userRow = self.userW[userInd, :]
        movieRow = self.movieW[movieInd, :].T
        if self.userFlag[userInd] == 0: userRow = self.userAvg
        if self.movieFlag[movieInd] == 0: movieRow = self.movieAvg.T
        return np.dot(userRow, movieRow)
    
    def evaluate(self, dataset: myDataset) -> float:
        """
        input: test dataset
        output: loss avg on current test set
        """
        loss_sum = 0
        for userInd, movieInd, rating in dataset:
            userInd, movieInd = int(userInd), int(movieInd)
            pred = self.predict(userInd, movieInd)
            loss_sum += calc_loss(rating, pred)
        return loss_sum / len(dataset)
