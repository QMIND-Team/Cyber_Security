#Features extracted using features.py from ember

One common function used in this is from sklearn, specifically skleanr.feature_extraction.FeatureHasher (imported just as FeatureHasher in ember)
I do not fully understand how this function works, and do not have time to fully investigate it at the moment. A lot of mysteries will be solved if someone learns how it works
 
With ALL of the classes, there are two main methods, one that will return a human-readable list of what is going on, and the other rerturns the actual vector of the data (few exeptions). FeatureHasher is the go-to way the authors used to translate between.

###class PEFeatureExtractor

This class is the control for the module. The only one that is actually called from an outside source/file. What this class does is initialize all the other classes within it, then passes the binary file through each of the others to get the information.

#####feature_vector(self,bytez)
the particular function from the class that is being called and returns the 2381 dim we are use to

###class ByteEntropyHistogram/Byte Histogram
#####This returns 512 of the total values

"Bin values of a two-dimensional byte entropy histogram that models the file's distribution of bytes"
What endgameinc did was create a 1024 byte 'window' over the input binary file, and move it along in steps, computing a base-2 entropy for each step, and each individual byte occurence in the window (how often each particular value for a byte occured)
Computer a maxtix from gathered information, that is 8x256, and concatenate it into a single 256 vector that is returned. 
I do not know the real distinction from the two, but this is how they work.
#####This class is almost certian to be of no use for us, I do not think it would be possible to recover any information from this to put back into the file

###class SectionInfo
" Information about section names, sizes and entropy.  Uses hashing trick to summarize all this section info into a feature vector."
It essentially takes the information from the section header, and displays creates it into a vector. This might be something good, as a stretch goal we might be able to reverse engineer it to get out a new section header.

from leif, takes advantage of lief.sections

###class ImportsInfo

As one might expect, this class returns all the imports from the PE has. Takes the name of the import as the key to a dictionary, and all the (methods?) of that import as the values related to that key, represented in a list. Not 100% sure what the other values are, and I have no idea how to find out what they mean. 

###class ExportsInfo
"Information about exported functions. Note that the total number of exported functions is contained in GeneralFileInfo."
uses lief.exported_functions to simply add all the exported functions from the PE to a list. 
Exported functions are functions in the PE that can be called by other modules within it.

###class GeneralFileInfo
{
'size': len(bytez),
'vsize': 0,
'has_debug': 0,
'exports': 0,
'imports': 0,
'has_relocations': 0,
'has_resources': 0,
'has_signature': 0,
'has_tls': 0,
'symbols': 0
}

change out the 0 for a call of leif.<key> and that is what this section does. Just takes it all into account, as the name suggests its general info
Very useful from the standpoint of finding if the PE is malicious or not, I do not think it will be particularily helpful for reconstructing the files. Some of the inputs might be, but not all

###class HeaderFileInfo

This class is very similar to the GeneralFileInfo, but instead takes all the information from the file header. Some stuff from the optional header is displayed here. Not everything, and I do not know the reasoning behind what is taken/what is not.

###class StringExtractor

I am pretty sure... not completely... that this just searches for strings within the PE. you know if you open up a PE in notepad, it is mostly randomness, but there are some words? It looks like this function looks for those, but I am not completely sure why/if it actually does.

Anyway, I do not tink this will be at all useful in the reconstruction of PE files.

###class DataDirectories

There is an additional area under optional header when viewing PE files via CFF explorer. This returns the values of each item from this table
