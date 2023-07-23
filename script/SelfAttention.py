from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from tensorflow.keras import backend as K

import tensorflow as tf
# from SpectralNormalizationKeras import ConvSN2D

def hw_flatten(x) :
        x_shape = K.shape(x)
        return K.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]
    
class SelfAttention(Layer):
    def __init__(self, filters, **kwargs):
        self.dim_ordering = K.image_data_format()
        self.filters = filters
        
        super(SelfAttention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer='zeros', trainable=True)
        
        super(SelfAttention, self).build(input_shape)  # Be sure to call this at the end
        
    def call(self,x):
        
        assert(len(x) == 4)
        img = x[0]
        f = x[1]
        g = x[2]
        h = x[3]
        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
        
        beta = K.softmax(s)  # attention map
        
        o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]        
        o = K.reshape(o, shape=[K.shape(img)[0], K.shape(img)[1], K.shape(img)[2], self.filters])  # [bs, h, w, C]
        img = self.gamma * o + img
        
        return img
    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
    def get_config(self):
        config = {'filters': self.filters}
        base_config = super(SelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
