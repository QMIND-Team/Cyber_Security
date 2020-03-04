# Features extracted using features.py from ember

features.py from ember works in two main stages: raw_features and process_raw_features.
Raw_features goes directly into the PE files and extracts information of what is going on. A lot of the features being used in this dataset have to do with more human-readable variables, such as file imports, exports, and the file header. raw_features will get this information from the files.

After raw_features has been run, process_raw_features is called. This method will take the information gained from raw_features, and convert it into floats, that can then be returned and are used as the values that are sent into the neural network. The method uses sklearn.feature_extraction.FeatureHasher to ensure the returns from process_raw_features is unique for different inputs, but the values are created in a logical and consistent way


### class ByteEntropyHistogram/Byte Histogram

##### This returns 512 of the total values

"Bin values of a two-dimensional byte entropy histogram that models the file's distribution of bytes". 
What this function was designed to do is create a 1024 byte 'window' over the input binary file, and move it along in steps, computing a base-2 entropy for each step, and each individual byte occurence in the window (how often each particular value for a byte occured)
Computer a maxtix from gathered information, that is 8 by 256, and concatenate it into a single 256 vector that is returned. 
I do not know the distinction from the two, but they both work similarily.

##### When we update the features we are using to try and reconstruct the files, this is likely not one we will be needing. 

### class SectionInfo

" Information about section names, sizes and entropy.  Uses hashing trick to summarize all this section info into a feature vector."
It takes the information from the section header and changes it into a vector using FeatureHasher.

This class will be very important for reconstructing a file, the sectionInfo is a key part of PE files.

### class ImportsInfo

This class returns all the imports from the PE has. Takes the name of the import as the key to a dictionary, and all the methods from each import as the values to the key, represented in a list. Values are specific pieces of information required for the import

### class ExportsInfo

"Information about exported functions. Note that the total number of exported functions is contained in GeneralFileInfo."
uses lief.exported_functions to add all the exported functions from the PE to a list. 
Exported functions are functions in the PE that can be called by other modules within it.

### class GeneralFileInfo

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

change out the 0 for a call of <leif.key> for any PE file, and it will display the information this function returns. It takes a lot of information from the file into account, and returns it all to be processed by the neural network. 
This would be very useful from the standpoint of finding if the PE is malicious or not, but would not be as useful for the reconstruction of PE files.

### class HeaderFileInfo

This class is very similar in structure to GeneralFileInfo, but it takes information from the file header/optional header only.

### class DataDirectories

This function takes the information from the data directories subsection under optional header in PE files. Takes the infromation, and directly returns it as a vector that can be processed.
