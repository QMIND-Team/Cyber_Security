import ember
import os 
import hashlib
from virus_total_apis import PublicApi as VirusTotalPublicApi

API_KEYS = ['b48590c8ff039b4ba923cbd2c8595bdcd6f568fabe5cf6a9f9533d033d6ef164',
           'f37176ea050c75200afa08ecfe1595211293df935ddac552f0fb28ad227c615f',
           'aee64bbcebf62c405a860a4967e40591b2ea0db9e7fd7acddb10ead3da3f6841',
           '0c19d50f19d8ccda36ceaf8512df723fe78ff473dbec4407003d0eb6fd3502b7',
           '89026d4458f7e0ab9aca686ec0b9576c91a51d49dff30c63e732b72fbfa8fcd6']

def massTest(folder,keys):
    pass

def singleTest(file,key):
    pass

if __name__=="__main__":

    EICAR = "X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*".encode('utf-8')
    EICAR_MD5 = hashlib.md5(EICAR).hexdigest()

    vt = VirusTotalPublicApi(API_KEY)

    response = vt.get_file_report(EICAR_MD5)