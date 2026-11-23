import tensorflow as tf
from config import NUM_CLASSES, HEIGHT, WIDTH

def decoder_block(input_tensor,skip_tensor,num_filters):
    x=tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(input_tensor)
    x = tf.keras.layers.Concatenate()([x, skip_tensor])
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(num_filters, (3, 3), padding="same", activation="relu")(x)
    return x

def make_Unet_with_resnet50():
    input_tensor=tf.keras.layers.Input(shape=(HEIGHT, WIDTH, 3))
    base_model=tf.keras.applications.ResNet50(input_tensor=input_tensor,include_top=False,weights="imagenet")
    layer_names=[
        "conv1_relu", #256*256
        "conv2_block3_out", #128*128
        "conv3_block4_out", #64x64
        "conv4_block6_out", #32*32
    ]
    skip_output=[base_model.get_layer(layer_name).output for layer_name in layer_names]
    bridge_output=base_model.get_layer("conv5_block3_out").output
    encoder_model=tf.keras.Model(inputs=input_tensor,outputs=skip_output+[bridge_output])

    s1,s2,s3,s4,bridge=encoder_model(input_tensor)
    #256,128,64,32

    #16*16 to 32*32
    d1=decoder_block(bridge,s4,512)
    #32*32 to 64
    d2=decoder_block(d1,s3,256)
    #64 to 128
    d3=decoder_block(d2,s2,128)
    #128 to 256
    d4=decoder_block(d3,s1,64)
    #256 to 512
    x=tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(d4)
    outputs=tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation="softmax",dtype=tf.float32)(x)

    model=tf.keras.Model(inputs=input_tensor,outputs=outputs,name="ResNet50_uNet")
    return model

if __name__ == "__main__":
    model = make_Unet_with_resnet50()
    model.summary()



