import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory

IMG = (128,128)
input_shape = (128,128,3)
train_data_dir = sys.argv[1]
btch_size=16

train_datagen = image_dataset_from_directory(
    train_data_dir, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=btch_size, image_size=IMG, seed=2000, validation_split=0.15, subset='training', follow_links=False
)

val_datagen = image_dataset_from_directory(
    train_data_dir, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=btch_size, image_size=IMG, seed=2000, validation_split=0.15, subset='validation',
    follow_links=False)




# pre_model = applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
pre_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
# pre_model = applications.VGG19(include_top=False, weights='imagenet', input_shape=input_shape)

# pre_model = 

pre_model.trainable=False

#      VGG19 fine tune
"""for layer in pre_model.layers[:16]:
	layer.trainable=False
"""

#	VGG16 fine tune
"""
for layer in pre_model.layers[:15]:
	layer.trainable= False
"""

print('Model loaded.')

model = Sequential()

model.add(Flatten(input_shape=pre_model.output_shape[1:]))

model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu', name='1stlay'))

model.add(Dropout(0.25))

model.add(Dense(512, activation='relu', name='2ndlay'))

# model.add(MaxPooling2D((2,2), strides= (2,2), name='maxpool2'))

# model.add(Dropout(0.2))

# model.add(Dense(512, activation='relu', name='3rdlay'))

# model.add(Dropout(.2))

model.add(Dense(2, activation='softmax'))

model = Model(inputs=pre_model.input, outputs=model(pre_model.output))

print(len(train_datagen))
print(model.summary())
model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(train_datagen, steps_per_epoch = len(train_datagen),
                         epochs = 8,
                         validation_data = val_datagen,
                         validation_steps = len(val_datagen))


model.save(sys.argv[2])
