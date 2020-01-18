"""
Use for determining if a file is a PE or not, and making sure all files are within a specific size range. 
First argument is the path you want to find PE files from,
second argument is the path you want to move PE files to.
Make sure the file for the path you are moving to already exists
before running program.
"""
import lief
import sys,os,shutil
from os import path

def Isolate_PE_From_Files(folder):
    os.mkdir(folder+"\\badFiles")
    for f in os.listdir(folder):
        """
        Try and except around this entire block are for abnormal cases
        where there is something wrong with the file reading that happen too rarely
        for me to know what is actually happening. It will print out what the error was,
        leave the file alone, and you will have to figure out if the file is good or not.
        Don't worry, very rare this will happen, from sample of about 1000 it happened once.
        """
        try:
            binary = lief.parse(folder+"\\"+f)
            #none if the file isn't a PE file, or a file format that can be read
            if type(binary) == type(None):
                os.remove(folder+"\\"+f)
                continue
            
            header = binary.header#if it can be read, guarnteed to have a header
            try:
                signature = header.signature
            except AttributeError:
                #this error means the file does not have a signature, and PE files MUST have one
                os.remove(folder+"\\"+f)
                continue
            
            #checking if the file signature is ascii for PE, which it will be in a proper PE file
            if signature[0:2] == [0x50,0x45]:
                pass
            else:
                os.remove(folder+"\\"+f)
                
        except Exception as e:
            shutil.move(folder+"\\"+f,folder+"\\badFiles\\"+f)
            print(e)

def fileSizeCheck(folder,minSize,maxSize):
    for f in os.listdir(folder):
        size = os.path.getsize(folder+"\\"+f)
        if size > maxSize or size < minSize:
            os.remove(folder+"\\"+f)

def help():
    print("FILE MUST BE RUN USING PYTHON 3.6 WITH LIEF INSTALLED. (python 3.7 may work if lief is updated for it.")
    print("Correct calling for file is:\n")
    print("findPE.py sizing [min file size] [max file size] [file folder]")
    print("min file size and max file size is in bytes")
    print("If you wish to ensure all files are within a specific size parameter\n")
    print("OR\n")
    print("findPE.py findPE [file folder]")
    print("If you wish to ensure all files within are in a PE format.")

if __name__ == "__main__":
    #get arguments from command line
    arguments = sys.argv

    if len(arguments) < 3:
        help()
        exit()

    if arguments[1] == 'sizing':
        if len(arguments) != 5:
            help()
            exit()
        if not path.isdir(arguments[4]):
            print("\nInvalid Path to folder\n\n")
            exit()
            
        fileSizeCheck(arguments[4],int(arguments[2]),int(arguments[3]))
    
    elif arguments[1] == 'findPE': 
        folder = arguments[2]
        #check to make sure starting path works
        if not path.isdir(folder):
            print("\nInvalid Path to folder\n\n")
            exit()

        Isolate_PE_From_Files(folder)
