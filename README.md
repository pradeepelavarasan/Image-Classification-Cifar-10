# Image Classification on Cifar 10 with limited parameters and epochs

In this project we are taking an existing model (base network) with the below details and trying to improve as per latest CNN standards:
- Number of parameter = ~1.2 million parameter
- Validation accuracy = 81.16
- No. of epochs = 50

The new customer model results are:
- Number of parameter = ~90 thousand parameters
- Validation accuracy = 82.39
- No. of epochs = 50

Note-> we have not considered image documented in this model and will be covered in other projects. Please feel free to try image augmentation to further improve your validation accuracy

## Custom Model
### Annotated with output size and receptive field calculation at every layer

model = Sequential()

model.add(SeparableConv2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(32, 32, 3), use_bias=False)) #Output size - 32*32*64 Output RF - 3
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(SeparableConv2D(128, 3, 3, activation='relu', border_mode='same', use_bias=False)) #Output size - 32*32*128 Output RF - 5
model.add(BatchNormalization())
model.add(Dropout(0.30))

#-------

model.add(Convolution2D(32, 1, 1, activation='relu', use_bias=False)) #Output size - 32*32*32 Output RF - 5
model.add(MaxPooling2D(pool_size=(2, 2))) #Output size - 16*16*32 Output RF - 6

#-------

model.add(SeparableConv2D(128, 3, 3, activation='relu', border_mode='same', use_bias=False)) #Output size - 16*16*128 Output RF - 10
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(SeparableConv2D(128, 3, 3, activation='relu',border_mode='same', use_bias=False)) #Output size - 16*16*128 Output RF - 14
model.add(BatchNormalization())
model.add(Dropout(0.30))

#-------

model.add(Convolution2D(64, 1, 1, activation='relu', use_bias=False)) #Output size - 16*16*64 Output RF - 14
model.add(MaxPooling2D(pool_size=(2, 2)))#Output size - 8*8*64 Output RF - 16

#-------

model.add(SeparableConv2D(128, 3, 3, activation='relu',border_mode='same', use_bias=False)) #Output size - 8*8*128 Output RF - 24
model.add(BatchNormalization())
model.add(Dropout(0.30))


model.add(SeparableConv2D(256, 3, 3, activation='relu',border_mode='same', use_bias=False)) #Output size - 8*8*256 Output RF - 32
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(SeparableConv2D(10, 1, 1, use_bias=False)) #Output size - 8*8*10 Output RF - 32
model.add(GlobalAveragePooling2D()) #Output size - 1*1*10
model.add(Activation('softmax'))




## Training and Validation Logs for 50 epochs

Epoch 1/50
48/48 [==============================] - 33s 692ms/step - loss: 1.7844 - acc: 0.3337 - val_loss: 2.6701 - val_acc: 0.2535

Epoch 2/50
48/48 [==============================] - 31s 648ms/step - loss: 1.3380 - acc: 0.5116 - val_loss: 2.6188 - val_acc: 0.3618

Epoch 3/50
48/48 [==============================] - 31s 641ms/step - loss: 1.1296 - acc: 0.5947 - val_loss: 1.8776 - val_acc: 0.4698

Epoch 4/50
48/48 [==============================] - 31s 643ms/step - loss: 1.0151 - acc: 0.6355 - val_loss: 1.8848 - val_acc: 0.4774

Epoch 5/50
48/48 [==============================] - 31s 649ms/step - loss: 0.9479 - acc: 0.6623 - val_loss: 1.3620 - val_acc: 0.5584

Epoch 6/50
48/48 [==============================] - 31s 641ms/step - loss: 0.8882 - acc: 0.6828 - val_loss: 1.4016 - val_acc: 0.5507

Epoch 7/50
48/48 [==============================] - 31s 648ms/step - loss: 0.8376 - acc: 0.7019 - val_loss: 1.2351 - val_acc: 0.5920

Epoch 8/50
48/48 [==============================] - 31s 649ms/step - loss: 0.8146 - acc: 0.7094 - val_loss: 1.0677 - val_acc: 0.6392

Epoch 9/50
48/48 [==============================] - 31s 640ms/step - loss: 0.7754 - acc: 0.7239 - val_loss: 1.0207 - val_acc: 0.6699

Epoch 10/50
48/48 [==============================] - 31s 643ms/step - loss: 0.7468 - acc: 0.7358 - val_loss: 0.9918 - val_acc: 0.6751

Epoch 11/50
48/48 [==============================] - 31s 641ms/step - loss: 0.7236 - acc: 0.7433 - val_loss: 0.9337 - val_acc: 0.6883

Epoch 12/50
48/48 [==============================] - 31s 639ms/step - loss: 0.6950 - acc: 0.7554 - val_loss: 0.9734 - val_acc: 0.6912

Epoch 13/50
48/48 [==============================] - 31s 640ms/step - loss: 0.6726 - acc: 0.7636 - val_loss: 0.8515 - val_acc: 0.7215

Epoch 14/50
48/48 [==============================] - 31s 640ms/step - loss: 0.6520 - acc: 0.7701 - val_loss: 0.8701 - val_acc: 0.7133

Epoch 15/50
48/48 [==============================] - 31s 638ms/step - loss: 0.6395 - acc: 0.7737 - val_loss: 0.8981 - val_acc: 0.7085

Epoch 16/50
48/48 [==============================] - 31s 638ms/step - loss: 0.6187 - acc: 0.7804 - val_loss: 0.8578 - val_acc: 0.7070

Epoch 17/50
48/48 [==============================] - 31s 641ms/step - loss: 0.6029 - acc: 0.7874 - val_loss: 0.6767 - val_acc: 0.7681

Epoch 18/50
48/48 [==============================] - 31s 636ms/step - loss: 0.5908 - acc: 0.7925 - val_loss: 0.7638 - val_acc: 0.7390

Epoch 19/50
48/48 [==============================] - 31s 639ms/step - loss: 0.5811 - acc: 0.7968 - val_loss: 0.7401 - val_acc: 0.7482

Epoch 20/50
48/48 [==============================] - 31s 643ms/step - loss: 0.5671 - acc: 0.8000 - val_loss: 0.7521 - val_acc: 0.7555

Epoch 21/50
48/48 [==============================] - 31s 641ms/step - loss: 0.5510 - acc: 0.8058 - val_loss: 0.6702 - val_acc: 0.7718

Epoch 22/50
48/48 [==============================] - 31s 638ms/step - loss: 0.5460 - acc: 0.8082 - val_loss: 0.6693 - val_acc: 0.7738

Epoch 23/50
48/48 [==============================] - 31s 643ms/step - loss: 0.5391 - acc: 0.8105 - val_loss: 0.6597 - val_acc: 0.7700

Epoch 24/50
48/48 [==============================] - 31s 644ms/step - loss: 0.5263 - acc: 0.8119 - val_loss: 0.6464 - val_acc: 0.7796

Epoch 25/50
48/48 [==============================] - 31s 649ms/step - loss: 0.5215 - acc: 0.8155 - val_loss: 0.7115 - val_acc: 0.7632

Epoch 26/50
48/48 [==============================] - 31s 650ms/step - loss: 0.5135 - acc: 0.8194 - val_loss: 0.6469 - val_acc: 0.7780

Epoch 27/50
48/48 [==============================] - 31s 642ms/step - loss: 0.4990 - acc: 0.8245 - val_loss: 0.6287 - val_acc: 0.7933

Epoch 28/50
48/48 [==============================] - 31s 647ms/step - loss: 0.5005 - acc: 0.8224 - val_loss: 0.6691 - val_acc: 0.7785

Epoch 29/50
48/48 [==============================] - 31s 647ms/step - loss: 0.4854 - acc: 0.8289 - val_loss: 0.6342 - val_acc: 0.7927

Epoch 30/50
48/48 [==============================] - 31s 645ms/step - loss: 0.4881 - acc: 0.8274 - val_loss: 0.6394 - val_acc: 0.7858

Epoch 31/50
48/48 [==============================] - 31s 644ms/step - loss: 0.4761 - acc: 0.8314 - val_loss: 0.6293 - val_acc: 0.7932

Epoch 32/50
48/48 [==============================] - 31s 641ms/step - loss: 0.4724 - acc: 0.8334 - val_loss: 0.5960 - val_acc: 0.7964

Epoch 33/50
48/48 [==============================] - 31s 637ms/step - loss: 0.4651 - acc: 0.8348 - val_loss: 0.6533 - val_acc: 0.7836

Epoch 34/50
48/48 [==============================] - 30s 635ms/step - loss: 0.4531 - acc: 0.8414 - val_loss: 0.5941 - val_acc: 0.8004

Epoch 35/50
48/48 [==============================] - 31s 636ms/step - loss: 0.4520 - acc: 0.8409 - val_loss: 0.6262 - val_acc: 0.7990

Epoch 36/50
48/48 [==============================] - 31s 640ms/step - loss: 0.4500 - acc: 0.8409 - val_loss: 0.6692 - val_acc: 0.7848

Epoch 37/50
48/48 [==============================] - 31s 638ms/step - loss: 0.4522 - acc: 0.8389 - val_loss: 0.5907 - val_acc: 0.8024

Epoch 38/50
48/48 [==============================] - 31s 641ms/step - loss: 0.4356 - acc: 0.8440 - val_loss: 0.6009 - val_acc: 0.8041

Epoch 39/50
48/48 [==============================] - 31s 638ms/step - loss: 0.4347 - acc: 0.8463 - val_loss: 0.5849 - val_acc: 0.8062

Epoch 40/50
48/48 [==============================] - 31s 638ms/step - loss: 0.4327 - acc: 0.8463 - val_loss: 0.6005 - val_acc: 0.8017

Epoch 41/50
48/48 [==============================] - 31s 640ms/step - loss: 0.4291 - acc: 0.8479 - val_loss: 0.6250 - val_acc: 0.7949

Epoch 42/50
48/48 [==============================] - 31s 647ms/step - loss: 0.4238 - acc: 0.8489 - val_loss: 0.6681 - val_acc: 0.7815

Epoch 43/50
48/48 [==============================] - 31s 650ms/step - loss: 0.4217 - acc: 0.8490 - val_loss: 0.5742 - val_acc: 0.8047

Epoch 44/50
48/48 [==============================] - 31s 645ms/step - loss: 0.4167 - acc: 0.8502 - val_loss: 0.6803 - val_acc: 0.7764

Epoch 45/50
48/48 [==============================] - 31s 647ms/step - loss: 0.4104 - acc: 0.8551 - val_loss: 0.7812 - val_acc: 0.7567

Epoch 46/50
48/48 [==============================] - 31s 650ms/step - loss: 0.4081 - acc: 0.8556 - val_loss: 0.6414 - val_acc: 0.7922

Epoch 47/50
48/48 [==============================] - 31s 648ms/step - loss: 0.4072 - acc: 0.8536 - val_loss: 0.6053 - val_acc: 0.7927

Epoch 48/50
48/48 [==============================] - 31s 646ms/step - loss: 0.4015 - acc: 0.8578 - val_loss: 0.6620 - val_acc: 0.7871

Epoch 49/50
48/48 [==============================] - 31s 643ms/step - loss: 0.3740 - acc: 0.8674 - val_loss: 0.5538 - val_acc: 0.8158

Epoch 50/50
48/48 [==============================] - 31s 645ms/step - loss: 0.3526 - acc: 0.8739 - val_loss: 0.5318 - val_acc: 0.8239

Model took 1546.88 seconds to train

Accuracy on test data is: 82.39
