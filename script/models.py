import numpy as np
from keras_self_attention import SeqSelfAttention
from keras_self_attention import SeqWeightedAttention as Attention
from spektral.utils.sparse import sp_matrix_to_sp_tensor 
from spektral.layers import GCNConv, GlobalAttentionPool, SortPool, TopKPool, GlobalSumPool, GlobalAttnSumPool, ARMAConv, APPNPConv


import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.layers import Input, Dropout, Flatten, LSTM
from tensorflow.keras import layers 
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as pp_input
from tensorflow.keras.optimizers.legacy import Adam, SGD, RMSprop


from RoiPoolingConvTF2 import RoiPoolingConv
from utils import getROIS, getIntegralROIS, crop, squeezefunc, stackfunc

# ################################################################################ #
# ############################# Model Definations ################################ #
# ################################################################################ #

def SR_GNN(pool_size = None, batch_size = None, ROIS_resolution = None, ROIS_grid_size = None, minSize = None, alpha = None, nb_classes = None):
    print('------------------------------------------------- Model Called : SR_GNN -------------------------------------')
    base_model = Xception(weights='imagenet', input_tensor=layers.Input(shape=(224,224,3)), include_top=False)
    base_out = base_model.output
    dims = base_out.shape.as_list()[1:]
    feat_dim = dims[2]*pool_size*pool_size
    base_channels = dims[2]

    x_final = base_out #or replace with SelfAttention
    full_img = layers.Lambda(lambda x: tf.image.resize(x,size=(ROIS_resolution, ROIS_resolution)), name='Lambda_img_1')(x_final) #Use bilinear upsampling (default tensorflow image resize) to a reasonable size
    print(' ')
    print('shape of full_image --> ', full_img.shape)
    print(' ')

    """Do the ROIs information and separate them out"""
    rois_mat =  getROIS(
        resolution = ROIS_resolution,
        gridSize = ROIS_grid_size, 
        minSize=minSize
        )
    # rois_mat = getIntegralROIS()
    # print(rois_mat)
    num_rois = rois_mat.shape[0]
    print("num_rois --> ", num_rois)


    roi_pool = RoiPoolingConv(pool_size=pool_size, num_rois=num_rois, rois_mat=rois_mat)(full_img)

    jcvs = []
    for j in range(num_rois):
        roi_crop = crop(1, j, j+1)(roi_pool)
        #roi_crop.name = 'lambda_crop_'+str(j)
        #print(roi_crop)
        lname = 'roi_lambda_'+str(j)
        x = layers.Lambda(squeezefunc, name=lname)(roi_crop) 
       
        x = layers.Reshape((feat_dim,))(x)
        jcvs.append(x)

    x = layers.Reshape((feat_dim,))(x_final)
    jcvs.append(x)
    jcvs = layers.Lambda(stackfunc, name='lambda_stack')(jcvs)
    jcvs=tf.keras.layers.Dropout(0.2) (jcvs)
    x = SeqSelfAttention(units=32, attention_activation='sigmoid', name='Attention')(jcvs) 
    x = layers.TimeDistributed(layers.Reshape((pool_size,pool_size, base_channels)))(x)
    x = layers.TimeDistributed(layers.GlobalMaxPooling2D(name='GMP_time'))(x)


    ######################

    x1 = layers.TimeDistributed(layers.Reshape((pool_size,pool_size, base_channels)))(jcvs)
    x1 = layers.TimeDistributed(layers.GlobalAveragePooling2D(name='GAP_time'))(x1)

    print(' ')
    print('shape of x1 --> ', x1.shape)
    print(' ')


    N = num_rois + 1
    gcn_input=layers.Input(shape=(N, 2048 ) )

    print(' ')
    print('shape of gcn_input --> ', gcn_input.shape, type(gcn_input))
    print(' ')


    A = np.ones((N,N), dtype='int')
    fltr = GCNConv.preprocess(A).astype('f4')
    A_in = Input(tensor=sp_matrix_to_sp_tensor(fltr), name='AdjacencyMatrix')

    print(' ')
    print('shape of fltr --> ', fltr.shape)
    print('shape of A_in --> ', A_in.shape)
    print(' ')

    channels = 1024
    gc1 = APPNPConv(channels, alpha=alpha, propagations=1, mlp_activation='sigmoid', use_bias=True) ([gcn_input, A_in])

    print(' ')
    print('shape of gcn1 --> ', gc1.shape, type(gc1))
    print(' ')

    gc = APPNPConv(channels, alpha=alpha, propagations=1, mlp_activation='sigmoid', use_bias=True) ([gc1, A_in])

    x3 = GlobalSumPool()(gc)
    x3 = tf.keras.layers.Dense(units=2048)(x3)
    tmp_model = Model(inputs=[gcn_input, A_in], outputs=x3, name='APPNP')
    x3 = tmp_model([x1, A_in])
    x2 = Attention(name='AttnWgt')(x)
    x2_bn =layers.BatchNormalization(name='BN1')(x2)
    x2 = tf.keras.layers.Activation('sigmoid') (x2_bn)
    x4 = tf.keras.layers.Dropout(0.2) (x3)
    multi = tf.keras.layers.Multiply()([x2, x4])
    sk = tf.keras.layers.add([x4, multi], name='add_sk')
    bn=layers.BatchNormalization(name='BN2')(sk)
    x4 = layers.Dense(nb_classes, activation='softmax')(bn)
    model = Model(inputs=[base_model.input, A_in], outputs=[x4])
    return model


##################################################################
######################## Model constructor #######################
##################################################################

def construct_model(name, **kwargs):
    if name == "srgnn":
        return SR_GNN(**kwargs)
    else:
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ Model defination not found! @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")