import pickle
import numpy as np


class myDataset:
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for i in range(0, self.data.shape[0]):
            yield self.data[i, :]
            
    def __len__(self):
        return self.data.shape[0]
            
    def getX(self):
        return self.data[:, :-1]
    
    def getY(self):
        return self.data[:, -1]


class CVdataloader:
    def __init__(self, cv, datapath, filename):
        self.cv = cv
        self.datapath = datapath
        self.filename = filename
        self.data = pickle.load(open(datapath + "/" + filename, "rb"))[:, :-1]
        np.random.shuffle(self.data)
        self.userNum = len(np.unique(self.data[:, 0]))
        self.movieNum = len(np.unique(self.data[:, 1]))
        self.seperates = list(range(0, self.data.shape[0], int((self.data.shape[0]-1)/self.cv)))
        self.seperates[-1] = self.data.shape[0]
        
    def __iter__(self):
        """
        return train dataset, test dataset for a cv round
        """
        for i in range(self.cv):
            testData = self.data[self.seperates[i]:self.seperates[i+1], :]
            trainData = np.vstack([self.data[0:self.seperates[i], :], self.data[self.seperates[i+1]:, :]])
            yield myDataset(trainData), myDataset(testData)


class NewUserLoader:
    def __init__(self, cv, datapath, filename, thresRatio):
        self.cv = cv
        self.datapath = datapath
        self.filename = filename
        self.data = pickle.load(open(datapath + "/" + filename, "rb"))[:, :-1]
        np.random.shuffle(self.data)
        self.userNum = len(np.unique(self.data[:, 0]))
        self.movieNum = len(np.unique(self.data[:, 1]))
        self.thres = int(self.userNum * thresRatio)
        
    def getTrain(self):
        return myDataset(self.data[self.data[:,0] <= self.thres])

    def getTest(self):
        return myDataset(self.data[self.data[:,0] > self.thres])


class NewItemLoader:
    def __init__(self, cv, datapath, filename, thresRatio):
        self.cv = cv
        self.datapath = datapath
        self.filename = filename
        self.data = pickle.load(open(datapath + "/" + filename, "rb"))[:, :-1]
        np.random.shuffle(self.data)
        self.userNum = len(np.unique(self.data[:, 0]))
        self.movieNum = len(np.unique(self.data[:, 1]))
        self.thres = int(self.movieNum * thresRatio)
        
    def getTrain(self):
        return myDataset(self.data[self.data[:,1] <= self.thres])

    def getTest(self):
        return myDataset(self.data[self.data[:,1] > self.thres])
