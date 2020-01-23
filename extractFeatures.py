import ember
import os
from ember.features import PEFeatureExtractor
import numpy as np
from tempfile import mkdtemp
from os import path


def vectorizePEs(inputPath):
    extractor = PEFeatureExtractor()
    nrows = 0
    bytez = []

    for file in os.listdir(inputPath):
        path = inputPath+"\\"+file
        with open(path,'rb') as f:
            bytez.append(f.read())
        
        nrows +=1
    print("Finished collecting all the binary information from files")
    vectorizedFeatures = np.array([extractor.feature_vector(bytez[i]) for i in range(len(bytez))])
    print("Finished extracting features from the binary")
    data = np.memmap(str(len(bytez))+"_files_features.dat",dtype='float32',mode='w+',shape=(nrows,extractor.dim))
    print("Finished creating the memmap for the data")
    data[:] = vectorizedFeatures[:]
    print("Finished mapping data to the memmap")

def readMemmap(file):
    #extracting data from a .dat file
    extractor = ember.PEFeatureExtractor()
    shape = (int(outPutFile.split("_")[0]),extractor.dim)
    data = np.memmap(outPutFile, dtype=np.float32, mode="r+", shape=shape)
    return data

if __name__ == "__main__":
    dataPath = "C:\Programming\Github_projects\Ember\extraData"
    outPutFile = "13476_files_features.dat" #only needed to use when extracting features from memmap
    #file name is created in part by num of files that went into it, and is needed to later
    #parse the file into the proper sizing from reading the memmap
    vectorizePEs(dataPath)