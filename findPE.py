"""
Use for determining if a file is a PE or not. 
First argument is the path you want to find PE files from,
second argument is the path you want to move PE files to.
Make sure the file for the path you are moving to already exists
before running program.
"""
import lief
import sys,os,shutil
from os import path

if __name__ == "__main__":
    #get arguments from command line
    arguments = sys.argv
    
    #extract starting file location
    try:
        fileLocation = arguments[1]
    except IndexError:
        print("Must add argument 1 as 'startPath' ")
        raise IndexError

    #extract file location to move files to
    try:
        endLocation = arguments[2]
    except IndexError:
        print("Must add argument 2 as 'endPath' ")
        raise IndexError

    #check to make sure starting path works
    if not path.isdir(fileLocation):
        print("Invalid Path")
        raise ValueError
    folder = path.join(fileLocation)

    #using lief to parse the file, making sure it is a PE
    for f in os.listdir(fileLocation):
        """
        Try and except around this entire block are for abnormal cases
        where there is something wrong with the file reading that happen too rarely
        for me to know what is actually happening. It will print out what the error was,
        leave the file alone, and you will have to figure out if the file is good or not.
        Don't worry, very rare this will happen, from sample of about 1000 it happened once.
        """
        try:
            binary = lief.parse(fileLocation+"\\"+f)
            if type(binary) == type(None):
                os.remove(fileLocation+"\\"+f)
                continue
            header = binary.header
            try:
                signature = header.signature
            except AttributeError:
                os.remove(fileLocation+"\\"+f)
                continue
            if signature[0:2] == [0x50,0x45]:
                shutil.move(fileLocation+"\\"+f,endLocation)
        except Exception as e:
            print(e)

