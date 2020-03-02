import lief,hashlib,os,findPE,shutil,ctypes,sys

def searchDir(currentDir,saveDir,ammount = 0):
    try:
        for f in os.listdir(currentDir):
            if ammount >= 10000:
                return
            if os.path.isdir(currentDir+"\\"+f):
                searchDir(currentDir+"\\"+f,saveDir,ammount)
            if os.path.isfile(currentDir+"\\"+f):
                good = findPE.fileSizeCheck(currentDir+"\\"+f,1,10000)
                if good:
                    good2 = findPE.Isolate_PE_From_Files(currentDir+"\\"+f)
                    if good2:
                        md5 = hashlib.md5((currentDir+"\\"+f).encode("utf-8")).hexdigest()
                        shutil.copy(currentDir+"\\"+f,saveDir+"\\Benign_"+md5)
                        
                        ammount +=1
                        if ammount % 100 == 0:
                            print("Found "+str(ammount))
    except Exception as e:
        print(e)

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if __name__ == "__main__":


    if not is_admin():
        # Make sure user has admin rights for the program to work
        ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)

        exit()
    currentDir = "C:\\Program Files"
    saveDir = "C:\\randomBenign"
    searchDir(currentDir,saveDir)

    currentDir = "C:\\Program Files(x86)"
    searchDir(currentDir,saveDir)

    currentDir = "C:\Windows"
    searchDir(currentDir,saveDir)
