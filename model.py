from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D
from keras import Model
from data_loader import DataLoader

def build_model():
	initial_model = InceptionV3(include_top=False, weights=None, pooling='max')
	x = initial_model.output
	x = Dense(1024, activation='relu')(x)
	x = Dense(1, activation='linear')(x)
	return Model(initial_model.input, x)

def train_model(model, d):
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=2,
                                 mode='auto')

	model.compile(loss='mean_squared_error', optimizer=Adam())

	model.fit_generator(d.generate_batch(),
						epochs=50,
						steps_per_epoch=d.n // d.batch_size + 1,
						max_queue_size=10,
                        callbacks=[checkpoint],
                        verbose=2)

	model.save('model.h5')

d = DataLoader('data/')
model = load_model('model-003.h5')
train_model(model, d)