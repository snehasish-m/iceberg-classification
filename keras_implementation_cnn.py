import numpy as np
import pandas as pd
import keras as k

#load data
#change directory accordingly
train = pd.read_json('../input/train.json')
train.inc_angle = train.inc_angle.replace('na',0)

def get_data(df, more_data):
    images = []
    for _, row in df.iterrows():  #iterate each row
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        
        #normalization
        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        band_2_norm = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        
        band_3 = band_1 / band_2  #additional band
        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        
        images.append(np.dstack((band_1_norm, band_2_norm, band_3_norm)))
        
    if more_data:    
        images = create_more_data(np.array(images))
    return np.array(images)   #debug-> convert to np.array
    
#idea from other kernels
def create_more_data(images):
    img_lr = []
    img_ud = []
    for i in range(0,images.shape[0]):
        ##left to right
        lr1 = np.flip(images[i,:,:,0], 0)  #for band1(index0)
        lr2 = np.flip(images[i,:,:,1], 0)  #for band2(index1)
        lr3 = np.flip(images[i,:,:,2], 0) #for band3(index2)
        img_lr.append(np.dstack((lr1, lr2, lr3)))  #dstack idea-> stackoverflow.com 
                                                   #np.concatenate isn't working :( 
        # mirror up-down
        ud1 = np.flip(images[i,:,:,0], 1)   #for band1(index0)
        ud2 = np.flip(images[i,:,:,1], 1)   #for band2(index1)
        ud3 = np.flip(images[i,:,:,2], 1)   #for band3(index2)
        img_ud.append(np.dstack((ud1, ud2, ud3)))

    return np.concatenate((images, np.array(img_lr), np.array(img_ud)))  #debug-> convert to np.array

x = get_data(train, more_data= True)
y = np.array(train['is_iceberg'])
y = np.concatenate((y, y, y)) #original + left-right + up-down (3 times)

#Building the model

model=k.models.Sequential()
model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3), activation='relu' ))
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.convolutional.Conv2D(128, kernel_size=(3, 3), activation='relu' ))
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.convolutional.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(k.layers.convolutional.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.Flatten())

model.add(k.layers.Dense(512))
model.add(k.layers.Activation('relu'))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(256))
model.add(k.layers.Activation('relu'))
model.add(k.layers.Dropout(0.2))

model.add(k.layers.Dense(1))
model.add(k.layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=k.optimizers.adam(0.001), metrics=['accuracy'])
#model.summary()

model.fit(x, y, batch_size=25, epochs=5, verbose=1, validation_split=0.05)


#testing and submitting
test = pd.read_json('../input/test.json')
test.inc_angle = train.inc_angle.replace('na',0)
test_X = get_data(test, more_data=False)
prediction = model.predict(test_X)
submission = pd.DataFrame({'id': test["id"], 'is_iceberg': prediction.reshape((prediction.shape[0]))})
submission.to_csv('submission.csv', index=False)