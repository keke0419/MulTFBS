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
```
 * example: PBMdatas/TF_1_Ar_pTH1739_HK
```

`encode.sh` is used to encode the sequence and transform it into one-hot and shape features.
```
 * Usage: bash encode.sh
```
