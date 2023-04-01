from keras.layers import Input, Conv2D, Conv2DTranspose, Concatenate
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.applications.efficientnet import EfficientNet
def build_ResNet():
    vgg_model = VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    return Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block3_conv4').output)

def build_EfficientNet():
    efficient_model = EfficientNet(include_top=False, weights='imagenet')
    efficient_model.trainable = False
    return Model(inputs=efficient_model.input, outputs=efficient_model.get_layer('block3_conv4').output)
