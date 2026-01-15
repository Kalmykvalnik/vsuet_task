import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# Загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Нормализация данных
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0

# One-hot кодирование меток
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Создание модели
model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

# Компиляция
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Вывод структуры сети
model.summary()

# Обучение
history = model.fit(
    x_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=1
)

# Оценка на тесте
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nТочность на тесте: {test_acc:.4f}")

# График точности
plt.plot(history.history["accuracy"], label="Точность на обучении")
plt.plot(history.history["val_accuracy"], label="Точность на валидации")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.legend()
plt.grid(True)
plt.show()
