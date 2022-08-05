from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Flatten, Dense, Conv2DTranspose, Lambda
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras import backend as K

from utils.util_layer import ClusteringLayer


"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _channel_shuffle(x, groups):
    """
    Channel shuffle layer

    # Arguments
        x: Tensor, input tensor of with `channels_last` or 'channels_first' data format
        groups: Integer, number of groups per channel

    # Returns
        Shuffled tensor
    """

    if K.image_data_format() == 'channels_last':
        height, width, in_channels = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, height, width, groups, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = K.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, groups, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]

    x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
    x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)
    x = Lambda(lambda z: K.reshape(z, later_shape))(x)

    return x


def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        alpha: Integer, width multiplier.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width
    cchannel = int(filters * alpha)

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    # x = _channel_shuffle(x, 4)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = Add()([x, inputs])

    return x


def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        alpha: Integer, width multiplier.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.

    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

    return x


def DSCAE(input_shape, filters=[32, 64, 128, 10], alpha=1.0):
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(inputs, first_filters, (5, 5), strides=(2, 2))
    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=2, n=1)
    x = _inverted_residual_block(x, 32, (3, 3), t=2, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 64, (3, 3), t=2, alpha=alpha, strides=2, n=2)
    x = Conv2D(128, kernel_size=(1, 1), padding='valid', strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(relu6)(x)

    en = Flatten()(x)
    en = Dense(filters[3], name='embedding')(en)
    
    if input_shape[1] % 8 != 0:
        de = Dense(filters[2]*int(input_shape[0]/32)*int(input_shape[1]/32 + 1), activation=relu6)(en)
        de = Reshape((int(input_shape[0] / 32), int(input_shape[1] / 32 + 1), filters[2]))(de)
    else:
        de = Dense(filters[2] * int(input_shape[0] / 32) * int(input_shape[1]/32), activation=relu6)(en)
        de = Reshape((int(input_shape[0] / 32), int(input_shape[1] / 32), filters[2]))(de)

    de = Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation=relu6, output_padding=(1, 1), name='deconv1_1')(de)
    identity = de
    de = Conv2DTranspose(filters[1], 3, strides=1, padding='same', activation=relu6, output_padding=(0, 0),name='deconv1_2')(de)
    de = Conv2DTranspose(filters[1], 3, strides=1, padding='same',activation=relu6,  name='deconv1_3')(de)
    de = Add()([de, identity])
    de = Conv2DTranspose(filters[1], 3, strides=2, padding='same', activation=relu6, output_padding=(1, 1), name='deconv2_1')(de)
    identity = de
    de = Conv2DTranspose(filters[1], 3, strides=1, padding='same', activation=relu6, output_padding=(0, 0),name='deconv2_2')(de)
    de = Conv2DTranspose(filters[1], 3, strides=1, padding='same',activation=relu6,  name='deconv2_3')(de)
    de = Add()([de, identity])
    de = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation=relu6, output_padding=(1, 0),
                         name='deconv3_1')(de)
    identity = de
    de = Conv2DTranspose(filters[0], 5, strides=1, padding='same', activation=relu6,  name='deconv3_2')(de)
    de = Conv2DTranspose(filters[0], 5, strides=1, padding='same', activation=relu6, output_padding=(0, 0),
                         name='deconv3_3')(de)
    de = Add()([de, identity])
    de = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', output_padding=(1, 1), name='deconv4_1')(de)
    identity = de
    de = Conv2DTranspose(input_shape[2], 5, strides=1, padding='same', name='deconv4_2')(de)
    de = Conv2DTranspose(input_shape[2], 5, strides=1, padding='same', name='deconv4_3')(de)
    de = Add()([de, identity])
    de = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', output_padding=(1, 1), name='deconv4_4')(de)

    model = Model(inputs, de)
    model.summary()
    return model


def DSClustering(input_shape, filters=[32, 64, 128, 10], alpha=1.0):
    inputs = Input(shape=input_shape)

    first_filters = _make_divisible(32 * alpha, 8)
    x = _conv_block(inputs, first_filters, (5, 5), strides=(2, 2))
    x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=2, n=1)
    x = _inverted_residual_block(x, 32, (3, 3), t=2, alpha=alpha, strides=2, n=2)
    x = _inverted_residual_block(x, 64, (3, 3), t=2, alpha=alpha, strides=2, n=2)
    x = Conv2D(128, kernel_size=(1, 1), padding='valid', strides=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation(relu6)(x)

    en = Flatten()(x)
    en = Dense(filters[3], name='embedding')(en)
    cluster = ClusteringLayer(10, name='clustering')(en)

    model = Model(inputs, cluster)
    model.summary()

    return model


if __name__ == '__main__':
    CAE = DSCAE((128, 156, 1))
    print(CAE.summary())
    cluster_model = DSClustering((128, 156, 1))
    print(cluster_model.summary())
