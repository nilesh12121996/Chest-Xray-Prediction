import sys
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import applications
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import load_model

IMG = (128,128)
input_shape = (128,128,3)
test_data_dir = sys.argv[1]
model_name = sys.argv[2]
btch_size=16

test_datagen = image_dataset_from_directory(
    test_data_dir, labels='inferred', label_mode='categorical',
    class_names=None, color_mode='rgb', batch_size=btch_size, image_size=IMG, follow_links=False
)

model = load_model(model_name)
Loss, score = model.evaluate_generator(test_datagen, steps=500, workers=1)
print("Accuracy: ",score," | Loss: ", Loss)
