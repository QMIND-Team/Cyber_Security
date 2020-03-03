import ember
import os,json,time,shutil
import hashlib
from virus_total_apis import PublicApi as VirusTotalPublicApi

API_KEYS = ['b48590c8ff039b4ba923cbd2c8595bdcd6f568fabe5cf6a9f9533d033d6ef164',
            'f37176ea050c75200afa08ecfe1595211293df935ddac552f0fb28ad227c615f',
            'aee64bbcebf62c405a860a4967e40591b2ea0db9e7fd7acddb10ead3da3f6841',
            '0c19d50f19d8ccda36ceaf8512df723fe78ff473dbec4407003d0eb6fd3502b7',
            '89026d4458f7e0ab9aca686ec0b9576c91a51d49dff30c63e732b72fbfa8fcd6']

def checkFileFromDIR(key,folder,file):
    md5 = file.split("_")[-1]
    response = key.get_file_report(md5)              #call api to get report on the file
    try:
        if response['response_code'] == 200:                    #response code 200 means everything went well
            if response['results']['total'] < 20:               #less than 20 antivirus checks on file means we will discard it, unsure how good it is
                print("Deleting a file")
                os.remove(folder+"\\"+file)

            elif response['results']['positives']/response['results']['total'] >0.5:            #50% or more say it is malicious
                shutil.move(folder+"\\"+file,folder+"\malicious\\"+file)
                with open(folder+"\malicious\metadata.json",'a') as f:
                    json.dump(response,f,sort_keys = True)
                    f.write("\n")
            elif response['results']['positives']/response['results']['total'] <=.05:           #5% or less say it is malicious, likely false positives from the few
                shutil.move(folder+"\\"+file,folder+"\\benign\\"+file)                          #calling it malicious, so we clasify as benign
                with open(folder+"\malicious\metadata.json",'a') as f:
                    json.dump(response,f,sort_keys = True)
                    f.write("\n")
            else:
                os.remove(folder+"\\"+file)                     #catch all removal for files that don't fit criteria
                print("Deleting a file")
        else:
            print(response['response_code'])                    #something went wrong, print out response code so we can find what it was
    except KeyError as e:
        print(response)

def massTest(folder, keys, virusShare = False):
    """
    Criteria for keeping a file as malicious:
    -at least 20 antiviruses must have been used to check the file
    -from the antiviruses, at least 50% of them must agree the file is malicous

    Criteria for calling a file benign/probably benign:
    -at least 20 antiviruses must have been used to check file
    -5% or less of the antiviruses think it is malicious
    """
    useKey = 0
    cycles = 0
    returned204 = 0#response code for too many requests
    vt = [VirusTotalPublicApi(key) for key in keys]
    
    #create path for sorted files to go
    if not os.path.isdir(folder+"\malicious"):
        os.mkdir(folder+"\malicious")
        f = open(folder+"\malicious\\metadata.json","w+")
        f.close()
    #make path for the files that most think are benign
    if not os.path.isdir(folder+"\\benign"):
        os.mkdir(folder+"\\benign")
        f = open(folder+"\\benign\\metadata.json","w+")
        f.close()

    #continuous loop until no more files to be sorted 
    while len(os.listdir(folder)) > 22:
        cycles +=1
        startTime = time.time()
        #5 keys * 4 calls a minute for the range(20) before a sleep() call
        for i in range(20):
            useKey += 1                                             #makes sure keys are nicely indexed through
            useKey %= 5
            file = os.listdir(folder)[2]                            #get next file from directory that is not the malicious/benign folders
            checkFileFromDIR(vt[useKey],folder,file)

        endTime = time.time()
        print(f"Sleeping after cycle {cycles}")
        time.sleep(62-int((endTime-startTime)))                     #sleep to make sure we don't overuse api
        print(f"Awake, starting cycle {cycles+1}")

    if len(os.listdir(folder)) > 2:                                 #the while loop will get pretty much everything sorted, but I made it to leave some
        for file in os.listdir(folder)[2:]:                         #files behind unsorted to make sure it doesn't break if all files are sorted middle of the for loop
            useKey += 1                                             #this just takes care of the remaining 20 or so files.
            useKey %= 5
            checkFileFromDIR(vt[useKey],folder,file)

def singleTest(file, key,virusShare = False):
    vt = VirusTotalPublicApi(key)
    #if file comes from virusShare.com, md5 is already part of the file name, so it doesn't have to be recalculated
    if virusShare:
        md5 = file.split("_")[-1]
    response = vt.get_file_report(md5)
    if response['response_code'] == 200:
        r = json.dumps(response,sort_keys=False)
        print(r)
    elif response['response_code'] == 204:#request rate exceded
        print("Unable to check the file with this API key, request rate exceded for either minute/day/month limit.")
        
def readMetadata(file):
    with open(file, 'r') as f:
        metadata = [json.loads(line) for line in f.readlines()]
    return metadata

if __name__=="__main__":
    #if you run this file, it will default to checking a bunch of files to see how malicious they are
    #and delete those that few antiviruses find as malicious
    
    #singleTest("C:\Programming\Github_projects\Ember\extraData\VirusShare_4ea73d1d9fd930aab23dba74515f6d23",API_KEYS[2],virusShare = True)
    
    massTest("D:\VirusShare_complete\VirusShare_complete",API_KEYS,True)

    #read in metaadata about what antiviruses said from the json files
    """
    metadataDIR = "C:\Programming\Github_projects\Ember\extraData\malicious\metadata.json"
    with open(metadataDIR,'r') as f:
        metadata = [json.loads(file) for file in f.readlines()]
    
    for data in metadata:
        print(json.dumps(data,sort_keys = False,indent = True))
        x = input()
    """
    

    
