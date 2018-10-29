import numpy as np
import tqdm
from utils import calc_loss
from dataProcessor import myDataset, CVdataloader


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
        loss_history = []		# avg loss of all samples for each epoch
        for i in tqdm.tqdm_notebook(range(self.epoch)):
            preds = []
            for batch_ind, (userInd, movieInd, rating) in enumerate(dataset):
                userInd, movieInd = int(userInd), int(movieInd)
                pred = self.predict()	# TODO
                preds.append(pred)
                #-----------------------------optimize function here--------------------------------

            loss_history.append(calc_loss(dataset.getY().flatten(), np.array(preds)).sum() / len(dataset))
        return loss_history
                    
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
