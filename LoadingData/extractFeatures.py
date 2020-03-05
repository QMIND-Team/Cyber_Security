import ember
from os import path
import os,shutil,json
from ember.features import PEFeatureExtractor
import numpy as np
from tempfile import mkdtemp 


def readMetadata(file):
    with open(file, 'r') as f:
        metadata = [json.loads(line) for line in f.readlines()]
    return metadata


def vectorizePEs(inputPath):
    extractor = PEFeatureExtractor()
    nrows = 0
    bytez = []

    for file in os.listdir(inputPath):
        path = inputPath+"\\"+file
        if '.json' not in file:
            with open(path, 'rb') as f:
                bytez.append(f.read())
            nrows += 1
        else:
            print("Not including the metadata file... for obvious reasons.")
        
    
    print("Finished collecting all the binary information from files")
    vectorizedFeatures = np.array([extractor.feature_vector(bytez[i]) for i in range(len(bytez))])
    print("Finished extracting features from the binary")
    data = np.memmap(str(len(bytez))+"_files_features.dat", dtype='float32', mode='w+', shape=(nrows, extractor.dim))
    print("Finished creating the memmap for the data")
    data[:] = vectorizedFeatures[:]
    print("Finished mapping data to the memmap")


def readMemmap(file):
    # extracting data from a .dat file
    dim1 = file.split("\\")
    dim1 = dim1[-1].split("_")
    extractor = ember.PEFeatureExtractor()
    shape = (int(dim1[0]), extractor.dim)
    data = np.memmap(file, dtype=np.float32, mode="r+", shape=shape)
    return data

def split_10k_files(filePath,newFilePath):
    """
    This function will split out 10000 files from a large ammount of data
    Could change to any number by changing the for loop, 10k was just the decided
    on ammount so it would be small enough to fit on github
    """
    extractor = ember.PEFeatureExtractor()
    metadata = readMetadata(filePath+'\metadata.json')
    for i in range(6820):
        try:                                    #index error would mean out of files left, so it just breaks the function where it is
            file = os.listdir(filePath)[1]      #has to be index 1 because index 0 is the .json file
            if not path.isdir(newFilePath):     #make new directory/metadata file if not already there
                os.mkdir(newFilePath)
                f = open(newFilePath+"\metadata.json","w+")
                f.close()
            #getting the response from virustotal for the particular file currently being moved
            segment = [i for i in metadata if i['results']['md5'] == file.split('_')[-1]][0]
            with open(newFilePath+"\metadata.json","a") as f:
                json.dump(segment,f,sort_keys = True)       #writing to the new .json file
                f.write("\n")
            
            shutil.move(filePath+"\\"+file, newFilePath)
        except IndexError:
            return
        if i%100 == 99:
            print(i+1,"iterations done")

if __name__ == "__main__":
    dataPath = "D:\VirusShare_complete\\viruses_unsorted"
    newFilePath = "D:\\VirusShare_complete\\10001Files"

    #outPutFile = "13476_files_features.dat" # only needed to use when extracting features from memmap
    # file name is created in part by num of files that went into it, and is needed to later
    # parse the file into the proper sizing from reading the memmap

    vectorizePEs(newFilePath)
    #split_10k_files(dataPath,newFilePath)
    #memmap = readMemmap("C:\\Users\\willm\\PycharmProjects\\Cyber_Security\\data\\10000_files_features.dat")
    #print(memmap)
    #print(len(memmap))
