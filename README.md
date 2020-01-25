# QMIND 2019-2020 Cyber_Security Team

### Our Goal
The goal of the Cyber Security team is to develop a Generative Adversarial Network (GAN) responsible for the creation
of malicious files that are capable of evading detection from antivirus software as well as a model that is capable 
of predicting if a file is malicious or benign.

The GAN is comprised of two main models that train against one another in attempts to beat one another. These two 
models are: the Generator - responsible for the creation of these malicious files to evade detection; and the 
Discriminator - responsible for classifying files as either malicious or benign.

As training progresses, the Generator will become increasingly good at creating malicious files that resemble benign 
files based on the previous predicted classification output by the Discriminator; While the Discriminator will 
simultaneously become increasingly good at detecting these files and classifying them as malicious.


### The Generator
The Generator model is responsible for creating malicious files to evade detection from antivirus software
##### Inputs
The Generator takes two tensors - first being a tensor containing vectorized features of malicious files in the shape
(None, 2381); the second input is a tensor of randomly generated noise within range 0 to 1 (*tf.random.uniform()*) of 
the same shape (None, 2381).
##### Output
The Generator model will output a singular *adversarial example* (tensor in the same shape as the inputs), of the
malicious vectorized features concatenated with the random generated noise.


### The Discriminator
The Discriminator model is responsible for making predictions on the classification of the adversarial examples output 
by the Generator as well as vectorized features of benign file vectorized features.
##### Inputs 
The Discriminator takes in a singular tensor of shape (None, 2381) containing vectorized features of either a benign 
file or an adversarial example output by the generator.
##### Outputs
The Discriminator model will output a single value as its prediction to whether or not the file is malicious or benign
(*0 indicating benign and 1 indicating malicious*).


### Data
##### Initial steps
The team began working with the *Endgame Malware Benchmark for Research (Ember)* (https://github.com/endgameinc/ember)
dataset, which consists of 1.1 million vectorized features taken from portable executable (PE) files.
##### Following steps
As the team developed models to train on the Ember dataset (not containing any PEs), the ability to revert the 
output adversarial examples to PE files was desired. The team then decided to pivot towards extracting vectorized
features from PE files ourselves, passing them through the GAN and with the use of the Lief python package will 
recreate the initial file based on the altered vectorized features. For the use of PE files, the team needed to pivot
to a new dataset in which VirusShare proved to be the most applicable to our uses.


## Team Members:
Will Macdonald

Ryan Saweczko

Will Coffell

Connor Chappell

Cameron Morrison


