from keras.models import Sequential, Model
from keras.layers import Convolution1D,RepeatVector, Embedding,Softmax, Add, MaxPooling1D, Dense,Lambda,Dropout,Permute, multiply, Input, concatenate, BatchNormalization, Activation, Flatten, Bidirectional, LSTM, GRU
from keras import regularizers
from keras import backend as K
from keras_pos_embd import PositionEmbedding, TrigPosEmbedding
import tensorflow as tf

#Pairwise attention module
def P_Attention(input, use_embedding='PositionEmbedding.MODE_ADD'):
    lstm_len = int(input.shape[1])
    lstm_dim = int(input.shape[2])
    if use_embedding == 'PositionEmbedding.MODE_CONCAT ':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=PositionEmbedding.MODE_CONCAT)
    elif use_embedding == 'PositionEmbedding.MODE_ADD':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=PositionEmbedding.MODE_ADD)
    elif use_embedding == 'TrigPosEmbedding.MODE_CONCAT ':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=TrigPosEmbedding.MODE_CONCAT)
    elif use_embedding == 'TrigPosEmbedding.MODE_ADD':
        pe_layer = PositionEmbedding(input_dim=lstm_len, output_dim=lstm_dim,
                                     mode=TrigPosEmbedding.MODE_ADD)
    fw_lstm_pe = pe_layer(input)

    dk = int(input.shape[2]) // 2
    dv = int(input.shape[2])

    Q = Dense(dk, use_bias=False)  # (bn,L,dk)
    K = Dense(dk, use_bias=False)  # (bn,L,dk)
    V = Dense(dv, use_bias=False)  # (bn,L,dv)
    QKt = Lambda(lambda x: tf.matmul(x[0], x[1]) / np.sqrt(dk))
    attention_score = Softmax(2, name='attention_score')
    attention_output = Lambda(lambda x: tf.matmul(x[0], x[1]))
    add_layer = Add()

    q_fw = Q(fw_lstm_pe)
    k_fw = K(fw_lstm_pe)
    v_fw = V(fw_lstm_pe)

    Kt = Permute((2, 1))  # (bn,dk,L)
    kt_fw = Kt(k_fw)
    QKt_fw = QKt([q_fw, kt_fw])  # (bn,L,L)

    attention_score_fw = attention_score(QKt_fw)  # (bn,L,L)
    attention_output_fw = attention_output([attention_score_fw, v_fw])  # (bn,L,dv)

    output = add_layer([input, attention_output_fw])
    return output


# build hybrid model with Single-attentional mechanism
def MulTFBS_SAtten(shape1=None, shape2=None,shape3=None,params=None, penalty=0.005):
    word_input = Input(shape=shape1,name='word2vec')
    X = Convolution1D(36, 7, activation='relu', padding='same')(word_input)
    X = Convolution1D(40, 11, activation='relu', padding='same')(X)
    #Single attention mechanism
    attention = Dense(1)(X)
    attention = Permute((2, 1))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=2), name='attention', output_shape=(34,))(attention)
    attention = RepeatVector(40)(attention)
    attention = Permute((2, 1))(attention)
    output = multiply([X, attention])

    output = Bidirectional(LSTM(24))(output)
    out_xin = Dropout(params['DROPOUT2'])(output)

    digit_input = Input(shape=shape2)
    X = Convolution1D(16, 13, activation='relu', padding='same')(digit_input)
    X = Convolution1D(32, 7, activation='relu', padding='same')(X)
    attention = Dense(1)(X)
    attention = Permute((2, 1))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=2), name='attention', output_shape=(35,))(attention)
    attention = RepeatVector(32)(attention)
    attention = Permute((2, 1))(attention)
    output = multiply([X, attention])
    output = Bidirectional(LSTM(32))(output)
    output = Dropout(params['DROPOUT1'])(output)
    share_model = Model(digit_input, output)

    main_input = Input(shape=shape2, name='sequence')
    out_main = share_model(main_input)

    auxiliary_input = Input(shape=shape3, name='shape')
    auxiliary_conv1 = Convolution1D(4, 1, activation='relu', padding='same', name='shape_conv')(auxiliary_input)
    out_aux = share_model(auxiliary_conv1)

    concat = concatenate([out_xin,out_main, out_aux], axis=-1)
    Y = BatchNormalization()(concat)
    Y = Dense(96, activation='relu', kernel_regularizer=regularizers.l2(penalty))(Y)
    Y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty))(Y)
    Y = Dropout(params['DROPOUT'])(Y)
    output = Dense(1)(Y)

    model = Model(inputs=[word_input, main_input, auxiliary_input], outputs=output)
    print(model.summary())
    return model

# build hybrid model with Pairwise-attentional mechanism
def MulTFBS_PAtten(shape1=None, shape2=None,shape3=None,params=None, penalty=0.005):
    digit_input = Input(shape=shape1)
    X = Convolution1D(36, 7, activation='relu', padding='same')(digit_input)
    X = Convolution1D(40, 11, activation='relu', padding='same')(X)
    output = P_Attention(X,use_embedding='PositionEmbedding.MODE_ADD')
    output = Bidirectional(LSTM(24))(output)
    out_xin = Dropout(params['DROPOUT2'])(output)

    all_input = Input(shape=shape2)
    X = Convolution1D(16, 13, activation='relu', padding='same')(all_input)
    X = Convolution1D(32, 7, activation='relu', padding='same')(X)
    output = P_Attention(X,use_embedding='PositionEmbedding.MODE_ADD')
    output = Bidirectional(LSTM(32))(output)
    output = Dropout(params['DROPOUT1'])(output)
    share_model = Model(all_input, output)

    main_input = Input(shape=shape2, name='sequence')
    out_main = share_model(main_input)

    auxiliary_input = Input(shape=shape3, name='shape')
    auxiliary_conv1 = Convolution1D(4, 1, activation='relu', padding='same', name='shape_conv')(auxiliary_input)
    out_aux = share_model(auxiliary_conv1)

    concat = concatenate([out_xin,out_main, out_aux], axis=-1)
    Y = BatchNormalization()(concat)
    Y = Dense(96, activation='relu', kernel_regularizer=regularizers.l2(penalty))(Y)
    Y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty))(Y)
    Y = Dropout(params['DROPOUT'])(Y)
    output = Dense(1)(Y)

    model = Model(inputs=[digit_input,main_input, auxiliary_input], outputs=output)
    print(model.summary())
    return model

# build hybrid model with only word2vec
def MulTFBS_onlyWord(shape1=None, params=None, penalty=0.005):
    word_input = Input(shape=shape1,name='word2vec')
    X = Convolution1D(36, 7, activation='relu', padding='same')(word_input)
    X = Convolution1D(40, 11, activation='relu', padding='same')(X)
    #Single attention mechanism
    attention = Dense(1)(X)
    attention = Permute((2, 1))(attention)
    attention = Activation('softmax')(attention)
    attention = Permute((2, 1))(attention)
    attention = Lambda(lambda x: K.mean(x, axis=2), name='attention', output_shape=(34,))(attention)
    attention = RepeatVector(40)(attention)
    attention = Permute((2, 1))(attention)
    output = multiply([X, attention])

    output = Bidirectional(LSTM(24))(output)
    out_xin = Dropout(params['DROPOUT2'])(output)

    Y = BatchNormalization()(out_xin)
    Y = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(penalty))(Y)
    Y = Dropout(params['DROPOUT'])(Y)
    output = Dense(1)(Y)

    model = Model(inputs=word_input, outputs=output)
    print(model.summary())
    return model