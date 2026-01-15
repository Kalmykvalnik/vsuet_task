import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

# Генерация данных: сумма двух случайных чисел
np.random.seed(42)
x_train = np.random.randn(1000, 2)  # 1000 примеров по 2 числа
y_train = x_train.sum(axis=1)  # Целевое значение - сумма

# Проверочная выборка
x_val = np.random.randn(200, 2)
y_val = x_val.sum(axis=1)

# Создание модели
model = keras.Sequential()
model.add(Dense(units=1, input_shape=(2,), activation="linear"))

# Компиляция модели
model.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(0.01))

# Обучение
history = model.fit(
    x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=0
)

# График обучения
plt.plot(history.history["loss"], label="Ошибка на обучении")
plt.plot(history.history["val_loss"], label="Ошибка на валидации")
plt.xlabel("Эпоха")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

# Проверка на тестовых данных
test_data = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 1.0]])
predictions = model.predict(test_data)
print("Результаты проверки:")
for i, (inp, pred) in enumerate(zip(test_data, predictions)):
    print(f"Вход: {inp}, Предсказано: {pred[0]:.4f}, Истина: {inp.sum()}")

# Веса сети
weights, bias = model.get_weights()
print(f"\nВеса: {weights.flatten()}")
print(f"Смещение: {bias[0]}")
