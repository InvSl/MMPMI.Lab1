from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()


# --- Приведение данных к подходящему формату
x_train = train_images.reshape((50000, 32*32*3))
x_train = x_train.astype('float32') / 255

x_test = test_images.reshape((10000, 32*32*3))
x_test = x_test.astype('float32') / 255

y_train = train_labels[:,0]  # поскольку метка класса является вектором
y_test = test_labels[:, 0]   

from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# --- Для проверки во время обучения
val_x = x_train[:10000]
partial_x_train = x_train[10000:]

val_y = y_train[:10000]
partial_y_train = y_train[10000:]


# --- Сетка
from keras import models, layers
model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_shape = (32*32*3,)))
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(128, activation = 'relu'))
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy',
              metrics = ['acc'])

history = model.fit(partial_x_train,
                    partial_y_train,
                    batch_size = 512,
                    epochs = 20,
                    validation_data = (val_x, val_y))


## --- Графики

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.subplots_adjust(wspace = 1)

#  loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

ax1.plot(epochs, loss, 'bo', label = 'Training loss')
ax1.plot(epochs, val_loss, 'b', label = 'Validation loss')
ax1.set_title('Training and validation loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

#   accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

ax2.plot(epochs, acc, 'bo', label = 'Training accuracy')
ax2.plot(epochs, val_acc, 'b', label = 'Vadidation accuracy')
ax2.set_title('Training and validation accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.show()


# --- Проверка на тестовой выборке
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)
