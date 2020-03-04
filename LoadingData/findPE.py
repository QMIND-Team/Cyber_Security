"""
Use for determining if a file is a PE or not, and making sure all files are within a specific size range. 
First argument is the path you want to find PE files from,
second argument is the path you want to move PE files to.
Make sure the file for the path you are moving to already exists
before running program.
"""
import lief
import sys, os, shutil
from os import path


def Isolate_PE_From_Files(file):
    """
    Try and except around this entire block are for abnormal cases
    where there is something wrong with the file reading that happen too rarely
    for me to know what is actually happening. It will print out what the error was,
    leave the file alone, and you will have to figure out if the file is good or not.
    Don't worry, very rare this will happen, from sample of about 1000 it happened once.
    """
    try:
        binary = lief.parse(file)
        # none if the file isn't a PE file, or a file format that can be read
        if type(binary) == type(None):
            return False
        
        header = binary.header  # if it can be read, guarnteed to have a header
        try:
            signature = header.signature
        except AttributeError:
            # this error means the file does not have a signature, and PE files MUST have one
            return False
        
        # checking if the file signature is ascii for PE, which it will be in a proper PE file
        if signature[0:2] == [0x50, 0x45]:
            pass
        else:
            return False
            
    except Exception as e:
        return False

    return True


def fileSizeCheck(file, minSize, maxSize):
    size = os.path.getsize(file)
    if size > maxSize or size < minSize:
        return False
    return True

def checkPE(folder):
    for f in os.listdir(folder):
        good = fileSizeCheck(folder+"\\"+f,1,10000)
        if not good:
            os.remove(folder+"\\"+f)
        else:
            good = Isolate_PE_From_Files(folder+"\\"+f)
            if not good:
                os.remove(folder+"\\"+f)
        

def help():
    print("FILE MUST BE RUN USING PYTHON 3.6 WITH LIEF INSTALLED. (python 3.7 may work if lief is updated for it.")
    print("When just using the file to find PE files, and delete the rest, run")
    print("findPE.py <folder path>")
    print("I updated the file, so this will automatically do everything we need to make")
    print("Sure the files are both less than 10k size, and actual PE files. Thanks!")
    print("For any confusion, typing findPE.py help will print this menu")


if __name__ == "__main__":
    # get arguments from command line
    arguments = sys.argv
    if len(arguments) != 2:
        help()
        exit()
    
    if (arguments[1]) == help():
        help()

    if not path.isdir(arguments[1]):
        print("\nInvalid Path to folder, does not exits\n\n")
    else:
        checkPE(argumetns[1])
