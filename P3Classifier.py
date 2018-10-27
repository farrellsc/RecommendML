import numpy as np
import tqdm
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
    
    def train(self, dataset: myDataset) -> list:
        """
        input: train dataset
        return: loss history by epoch as a list
        """
        loss_history = []
        for i in tqdm.tqdm_notebook(range(self.epoch)):
            preds = []
            for batch_ind, (userInd, movieInd, rating) in enumerate(dataset):
                userInd, movieInd = int(userInd), int(movieInd)
                pred = self.predict(userInd, movieInd)
                preds.append(pred)
                userG = 2 * self.lr * (rating - pred) * self.movieW[movieInd, :]
                movieG = 2 * self.lr * (rating - pred) * self.userW[userInd, :]
                self.userW[userInd, :] += userG
                self.movieW[movieInd, :] += movieG
            loss_history.append(self.calc_loss(dataset.getY().flatten(), np.array(preds)).sum() / len(dataset))
        return loss_history
                    
    def predict(self, userInd, movieInd):
        """
        make prediction for one sample
        """
        return np.dot(self.userW[userInd, :], self.movieW[movieInd, :].T)
    
    def evaluate(self, dataset: myDataset) -> float:
        """
        input: test dataset
        output: loss sum on current test set
        """
        loss_sum = 0
        for userInd, movieInd, rating in dataset:
            userInd, movieInd = int(userInd), int(movieInd)
            pred = self.predict(userInd, movieInd)
            loss_sum += self.calc_loss(rating, pred)
        return loss_sum / len(dataset)
    
    @staticmethod
    def calc_loss(true, pred):
        return (true - pred) ** 2
