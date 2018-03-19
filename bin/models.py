from imports import *


##############################################
####  GRU/LSTM  ##############################
##############################################

def GRU_LSTM_model(CuDNN, maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNN(128, return_sequences=True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


##############################################
####  CAPSULE  ###############################
##############################################

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def CAPSULE_model(maxlen, max_features, embed_size, embedding_matrix, rate_drop_dense, 
                 Num_capsule, Dim_capsule, Routings, gru_len):
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

#     x = Bidirectional(
#         GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
#         embed_layer)

    x = Bidirectional(
        CuDNNGRU(gru_len,return_sequences=True))(embed_layer)
    x = Dropout(rate_drop_dense)(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(rate_drop_dense)(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    return model


##############################################
####  DCNN     ###############################
##############################################

def DPCNN_model(maxlen, max_features, embed_size, embedding_matrix, spatial_dropout,
               filter_nr, filter_size, max_pool_size, max_pool_strides, dense_nr, dense_dropout):
    comment = Input(shape=(maxlen,))
    emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(comment)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    #we pass embedded comment through conv1d with filter size 1 because it needs 
    # to have the same shape as block output
    #if you choose filter_nr = embed_size (300 in this case) you don't have 
    # to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear')(emb_comment)
    resize_emb = PReLU()(resize_emb)

    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)

    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)

    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear')(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    output = add([block4, block3_output])
    output = GlobalMaxPooling1D()(output)
    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(6, activation='sigmoid')(output)

    model = Model(comment, output)

    model.compile(loss='binary_crossentropy', 
                optimizer=optimizers.Adam(),
                metrics=['accuracy'])
    return model


##############################################
####  DCNN     ###############################
##############################################


class CV_predictor():
    '''
    class to extract predictions on train and test set from tunned pipeline
    '''
    def __init__(self, get_model, x_train, y_train, x_test, 
                 n_splits, batch_size, epochs, col_names, 
                 model_kwargs):
        self.get_model = get_model
        self.cv = KFold(n_splits=n_splits, shuffle=True, random_state=1)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.scorrer = roc_auc_score
        self.train_predictions = []
        self.test_predictions = []
        self.score = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.col_names = col_names        
        self.model_kwargs = model_kwargs
#         self.maxlen, self.max_features, self.embed_size, self.embedding_matrix
    
    def predict(self):
        test_number = 1
        for train_i, valid_i in self.cv.split(self.x_train, self.y_train):
            model = self.get_model(**self.model_kwargs)
            x_train = self.x_train[train_i]
            y_train = self.y_train[train_i]
            x_valid = self.x_train[valid_i]
            y_valid = self.y_train[valid_i]
            for i in self.epochs:
                model.fit(x_train, y_train, epochs = 1, batch_size = self.batch_size)
                train_prediction = model.predict(x_valid, self.batch_size * 2)
                print (f'test_number: {test_number}, epoch: {i}, score: {self.scorrer(y_valid, train_prediction)}')
            test_prediction =  model.predict(self.x_test, self.batch_size * 2)
            self.train_predictions.append([train_prediction, valid_i])
            self.test_predictions.append(test_prediction)
            self.score.append(self.scorrer(y_valid, train_prediction))
            print (f"test_number: {test_number}, avg score: {self.score[-1]}")
            test_number += 1
        print (np.mean(self.score))
        self.train_predictions = (
            pd.concat([pd.DataFrame(data=i[0],index=i[1], columns=[self.col_names]) 
                       for i in self.train_predictions]).sort_index())
        self.test_predictions = pd.DataFrame(data=np.mean(self.test_predictions, axis=0), columns=[self.col_names])