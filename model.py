import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input

class DeepfakeDataGenerator(Sequence):
    def __init__(self, directory, batch_size=15, sequence_length=10, shuffle=True, augment=False, datagen=None):
        self.directory = directory
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.image_paths = self._get_image_paths()
        self.augment = augment
        self.datagen = datagen
        self.on_epoch_end()

    def _get_image_paths(self):
        image_paths = []
        for category in ['Celeb-real', 'Celeb-synthesis', 'YouTube-real']:
            cat_dir = os.path.join(self.directory, category)  # This will now point to 'processed_frames/Celeb-real', etc.
            for image in os.listdir(cat_dir):
                if image.endswith(('.jpg', '.png')):
                    image_paths.append((os.path.join(cat_dir, image), 1 if category == 'Celeb-synthesis' else 0))
        return image_paths

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size * self.sequence_length)))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size * self.sequence_length:
                                       (idx + 1) * self.batch_size * self.sequence_length]
        batch_x = []
        batch_y = []

        for i in range(0, len(batch_paths), self.sequence_length):
            sequence_x = []
            sequence_y = []

            for j in range(self.sequence_length):
                image_path, label = batch_paths[i + j]
                img = load_img(image_path, target_size=(128, 128))
                img_array = img_to_array(img) / 255.0
                if self.augment and self.datagen:
                    img_array = self.datagen.random_transform(img_array)
                sequence_x.append(img_array)

            batch_x.append(sequence_x)
            batch_y.append(label)

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_paths)


base_path = r"C:\Users\HP\OneDrive\Desktop\projects\deepfake\deepfake-data\processed_frames"
batch_size = 15
sequence_length = 10

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

all_data = DeepfakeDataGenerator(base_path, batch_size=batch_size, sequence_length=sequence_length)
train_size = int(0.8 * len(all_data.image_paths) / sequence_length)
train_paths, val_paths = all_data.image_paths[:train_size * sequence_length], all_data.image_paths[train_size * sequence_length:]

train_generator = DeepfakeDataGenerator(base_path, batch_size=batch_size, sequence_length=sequence_length, augment=True, datagen=datagen)
train_generator.image_paths = train_paths
val_generator = DeepfakeDataGenerator(base_path, batch_size=batch_size, sequence_length=sequence_length, shuffle=False)
val_generator.image_paths = val_paths


resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(128, 128, 3))


for layer in resnet.layers[:-10]:
    layer.trainable = False


model = Sequential([
    TimeDistributed(resnet, input_shape=(sequence_length, 128, 128, 3)),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    epochs=10,
                    validation_data=val_generator)

model.save('deepfake_detection_model_lstm.h5')

val_loss, val_accuracy = model.evaluate(val_generator)

print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
