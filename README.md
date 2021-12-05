# MulTFBS
MulTFBS：A deep learning framework with multi-features integrated for predicting transcription factor binding sites
## Requirements
 * Python=3.6
 * TensorFlow=1.2
 * Keras=2.1.3
 * biopython=1.79
 * gensim=3.8.3
## Data preparation
After downloading the data according to the link in the paper, the data should be trimmed first to ensure equal length of the data, and then the data should be placed in the `PBMdatas` directory for the convenience of subsequent operation and experiment.
 *  example: 
 ```
 PBMdatas/TF_1_Ar_pTH1739_HK/TF_1_Ar_pTH1739_HK
 ```
`encode.sh` is used to encode the sequence and transform it into one-hot and shape features.
 * Usage: 
 ```
 bash encode.sh
 ```
Run `word2vec_skipgram.py` to produce the word2vec word vector. The word vector is then indexed in `MulTFBS_run.py` to the word2vec features of the sequence.Note that you need to convert the data to `> `, and then place it in the `PBMdatas` directory.
*  example: 
 ```
 PBMdatas/TF_1_Ar_pTH1739_HK/TF_1_Ar_pTH1739_HK_1
 ```

## Train and test MulTFBS
Run the main code ` MulTFBS_run.py` to cross-validate the model at 5-fold, with 4-fold for training and 1-fold for testing.
* You can run different data by modifying the path of the input. 
