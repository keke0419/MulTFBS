# MulTFBS
MulTFBS: A deep learning framework with multi-channels for predicting transcription factor binding sites
## Requirements
 * Python=3.6,recommend installing Anaconda3
 * TensorFlow=1.2
 * Keras=2.1.3
 * Biopython=1.79
 * Gensim=3.8.3
 * Matplotlib=3.3.2
 * H5py,Hyperopt,Sklearn,Nltk
## Data preparation
The PBMdata used in this work can be downloaded from 
https://bitbucket.org/wenxiu/sequence-shape/get/2159e4ef25be.zip 
After that the data should be trimmed first to ensure equal length of the data, and then the data should be placed in the `PBMdatas` directory for the convenience of subsequent operation and experiment.
 *  Example: 
 ```
 PBMdatas/TF_1_Ar_pTH1739_HK/TF_1_Ar_pTH1739_HK
 ```
`encode.sh` is used to encode the sequence and transform it into one-hot and shape features.
 * Usage: 
 ```
 bash encode.sh
 ```
Run `word2vec_skipgram.py` to produce the Word2vec word vector. The word vector is then indexed in `MulTFBS_run.py` to the Word2vec features of the sequence. Note that before doing this you need to convert 66 original data to FASTA format for subsequent reading of the base sequence, and then place it in the `PBMdatas` directory.
*  Example: 
 ```
 PBMdatas/TF_1_Ar_pTH1739_HK/TF_1_Ar_pTH1739_HK_1
 ```
In the meantime, you need to generate the same form of integrated corpus to learn word vectors.
## Train and test MulTFBS
Run the main code ` MulTFBS_run.py` to cross-validate the model at 5-fold, with 4-fold for training and 1-fold for testing.
* You can run different data by modifying the path of the input. 
## Contact
If you have any help and problems, please contact Kexin: kexin1004_0919@163.com
