# model_unet.py

from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x

def build_unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    bn = conv_block(p4, 512)

    u1 = layers.UpSampling2D()(bn)
    u1 = layers.Concatenate()([u1, c4])
    c5 = conv_block(u1, 256)

    u2 = layers.UpSampling2D()(c5)
    u2 = layers.Concatenate()([u2, c3])
    c6 = conv_block(u2, 128)

    u3 = layers.UpSampling2D()(c6)
    u3 = layers.Concatenate()([u3, c2])
    c7 = conv_block(u3, 64)

    u4 = layers.UpSampling2D()(c7)
    u4 = layers.Concatenate()([u4, c1])
    c8 = conv_block(u4, 32)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c8)

    return models.Model(inputs, outputs)
