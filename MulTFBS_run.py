from nltk import bigrams
from os.path import join, abspath, dirname, exists
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Bio import SeqIO
from gensim.models import Word2Vec
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
from utils import *
from models import *

#Word2vec feature index
texts_1 = []
for index, record in enumerate(SeqIO.parse('./PBMdatas/TF_2_Dbp_pTH3831_HK/TF_2_Dbp_pTH3831_HK_1.txt', 'fasta')):
    tri_tokens = bigrams(record.seq)
    temp_str = ""
    for item in ((tri_tokens)):
        #print(item),
        temp_str = temp_str + " " +item[0] + item[1]
        #temp_str = temp_str + " " +item[0]
    texts_1.append(temp_str)
seq_1=[]
stop = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
for doc in texts_1:
    doc = re.sub(stop, '', doc)
    seq_1.append(doc.split())

wv2 = './word2vec/all_word2vec_2.50.model'
wvz= Word2Vec.load(wv2)
xtrain_1 = []
for i in seq_1:
    a = wvz.wv[i]
    xtrain_1.append(a)
word_data = np.array(xtrain_1)

file_path = dirname(abspath(__file__))

# Introduce other features
datadir = './PBMdatas/TF_2_Dbp_pTH3831_HK/data'
name = (datadir).split('/')[-2]
print('working on %s now' % name)

print('using DNA sequences and MPRH shape features.')
seq_path_train = datadir + '/train.hdf5'
shape_path_train = datadir + '/train_MPRH.hdf5'
with h5py.File(seq_path_train, 'r') as f1, h5py.File(shape_path_train, 'r') as f2:
    seqs_data_train = np.asarray(f1['data'])
    intensity_train = np.asarray(f1['intensity'])
    shape_data_train_1 = np.array(f2['shape'])
    shape_data_train = np.array(shape_data_train_1)
    for i in range(shape_data_train.shape[0]):
        shape_data_train[i, :, :] = (shape_data_train[i, :, :] - np.mean(shape_data_train[i, :, :])) / np.std(
            shape_data_train[i, :, :])
    seqs_num = seqs_data_train.shape[0];
    seqs_len = seqs_data_train.shape[1];
    seqs_dim = seqs_data_train.shape[2]
    print('there are %d seqences, each of which is a %d*%d array' % (seqs_num, seqs_len, seqs_dim))
    input_shape2 = (seqs_len, seqs_dim)
    shape_num = shape_data_train.shape[0];
    shape_len = shape_data_train.shape[1];
    shape_dim = shape_data_train.shape[2];
    print('there are %d shape sequences, each of which is a %d*%d array' % (shape_num, shape_len, shape_dim))
    input_shape3 = (shape_len, shape_dim)
w_num = word_data.shape[0];
w_len = word_data.shape[1];
w_dim = word_data.shape[2];
print('there are %d word2vec sequences, each of which is a %d*%d array' % (w_num, w_len, w_dim))
input_shape1 = (w_len, w_dim)
# else:
#     print('invalid command!', file=sys.stderr);
#     sys.exit(1)

assert w_num == seqs_num, "the number of them must be consistent."
assert seqs_len == shape_len, "the length of them must be consistent."

# k-folds cross-validation
indices = np.arange(w_num)
np.random.shuffle(indices)
seqs_data_train = seqs_data_train[indices]
intensity_train = intensity_train[indices]
shape_data_train = shape_data_train[indices]
word_train = word_data[indices]


train_ids, test_ids, valid_ids = Id_k_folds(w_num, 5, 0.125)
PCC = [];R2 = []

model_name = 'model_result'
if not exists(file_path + '/%s/%s' % (model_name, name)):
   print('Building ' + file_path + '/%s/%s' % (model_name, name))
   os.makedirs(file_path + '/%s/%s' % (model_name, name))
f_params = open(file_path + '/%s/%s/params.txt' % (model_name, name), 'w')
for fold in range(5):
    x_train = seqs_data_train[train_ids[fold]]
    shape_train = shape_data_train[train_ids[fold]]
    w_train = word_train[train_ids[fold]]
    y_train = intensity_train[train_ids[fold]]

    x_valid = seqs_data_train[valid_ids[fold]]
    shape_valid = shape_data_train[valid_ids[fold]]
    w_valid = word_train[valid_ids[fold]]
    y_valid = intensity_train[valid_ids[fold]]

    x_test = seqs_data_train[test_ids[fold]]
    shape_test = shape_data_train[test_ids[fold]]
    w_test = word_train[test_ids[fold]]
    y_test = intensity_train[test_ids[fold]]


    history_all = {}
    for params_num in range(12):
        params = RandomSample()
        print("the {}-th paramter setting of the {}-th fold is {}".format(params_num, fold, params),
                    file=f_params)

        print('Building model...')
        model = MulTFBS_SAtten(input_shape1,input_shape2,input_shape3, params)
        checkpointer = ModelCheckpoint(filepath=file_path + '/%s/%s/params%d_bestmodel_%dfold.hdf5'
                                                        % (model_name, name, params_num, fold),
                                               monitor='val_loss', verbose=1, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

        print('Training model...')
        # myoptimizer = Adadelta(epsilon=params['DELTA'], rho=params['MOMENT'])
        model.compile(loss='mean_squared_error', optimizer='adam')
        History = model.fit([w_train,x_train,shape_train], [y_train], epochs=100, batch_size=300,
                                    shuffle=True,
                                    validation_data=([w_valid,x_valid,shape_valid], [y_valid]),
                                    callbacks=[checkpointer, earlystopper], verbose=2)
        history_all[str(params_num)] = History
    best_num = SelectBest(history_all, file_path + '/%s/%s/' % (model_name, name), fold, 'val_loss')

    PlotandSave(history_all[str(best_num)],file_path + '/%s/%s/figure_%dfold.png' % (model_name, name, fold), fold, 'val_loss')
    print("\n\n", file=f_params)
    f_params.flush()
    print('Testing model...')
    # load_model('')
    model.load_weights(file_path + '/%s/%s/params%d_bestmodel_%dfold.hdf5' % (model_name, name, best_num, fold))
    results = model.evaluate([w_test,x_test,shape_test], [y_test])
    print(results)
    y_pred = model.predict([w_test,x_test,shape_test], batch_size=300, verbose=1)
    y_pred = np.asarray([y[0] for y in y_pred])
    y_real = np.asarray([y[0] for y in y_test])
    with open(file_path + '/%s/%s/score_%dfold.txt' % (model_name, name, fold), 'w') as f:
       assert len(y_pred) == len(y_real), 'dismathed!'
       for i in range(len(y_pred)):
          print('{:.4f} {}'.format(y_pred[i], y_real[i]), file=f)

    print('Calculating R2...')
    pcc, r2 = ComputePCC(y_pred, y_real)
    PCC.append(pcc)
    R2.append(r2)

f_params.close()
print("the mean R2 is {}. the mean PCC is {}".format(np.mean(R2),np.mean(PCC)))
outfile = file_path + '/%s/%s/metrics.txt' % (model_name, name)
with open(outfile,'w') as f:
    for i in range(len(R2)):
        print("{:.4f} {:.4f}".format(R2[i], PCC[i]), file=f)
    print("{:.4f} {:.4f}".format(np.mean(R2),np.mean(PCC)), file=f)

