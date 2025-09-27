
""" Назначение: Реализация агента DQN и вспомогательных функций обучения """


import numpy as np
import random
from tensorflow.keras import models, layers, optimizers
from collections import deque


class DQNAgent:
    def __init__(self, state_size, action_size):
        """
        Конструктор агента DQN

        :param state_size: размер входного вектора состояния.
        :param action_size: размер выходного вектора действий.
        """

        # Размер входного состояния
        self.state_size = state_size
        # Размер выходного действия
        self.action_size = action_size
        # Буфер хранения опытных пар
        self.memory = deque(maxlen=2000)
        # Коэффициент дисконтирования будущих наград
        self.gamma = 0.99
        # Начальная вероятность случайного действия
        self.epsilon = 1.0
        # Степень затухания вероятности случайных действий
        self.epsilon_decay = 0.98
        # Минимальное значение эпсилона
        self.epsilon_min = 0.01
        # Скорость обучения
        self.learning_rate = 0.005
        # Строим нейронную сеть
        self.model = self._build_model()

    def _build_model(self):
        """
        Метод построения нейронной сети для агента.
        """

        model = models.Sequential([
            # Первый скрытый слой
            layers.Dense(
                24,
                input_dim=(self.state_size + 1),
                activation='relu'
            ),
            # Второй скрытый слой
            layers.Dense(24, activation='relu'),
            # Выходной слой
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            # Используем среднеквадратичную ошибку
            loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Запоминание опытной пары
        (состояние-действие-награда-новый_состояние-концовка).
        """

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Метод выбора действия.
        :param state: текущее состояние среды.
        :return: вектор действий (включить/выключить каждое устройство).
        """

        # Случайное действие с вероятностью epsilon
        if np.random.rand() <= self.epsilon:
            # Включаем/выключаем устройства и управляем аккумулятором
            return [random.choice([0, 1]) for _ in range(self.action_size)]

        # Прогнозируем Q-значения
        q_values = self.model.predict(np.array([state]), verbose=0)
        # Принятие решения на основе Q-значений
        return [1 if q > 0.5 else 0 for q in q_values[0]]

    def replay(self, batch_size):
        """
        Воспроизведение опыта для обучения.
        :param batch_size: размер минибатча для обучения.
        """

        # Берём случайный набор воспоминаний
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Цель равна полученной награде
            target = reward
            if not done:                 # Если период не завершён
                target += self.gamma * \
                    np.amax(self.model.predict(
                        np.array([next_state]), verbose=0)[0])
            # Прогнозируем Q-значения
            target_f = self.model.predict(np.array([state]), verbose=0)
            # Обновляем целевое значение для активных устройств
            for i, act in enumerate(action):
                target_f[0][i] = target if act == 1 else target_f[0][i]
            # Обучаемся на одной эпохе
            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:       # Эпсилон-декрементация
            self.epsilon *= self.epsilon_decay

    def save_model(self, filename):
        """
        Сохранение модели в файл.
        """
        self.model.save(filename)

    def load_model(self, filename):
        """
        Загрузка модели из файла.
        """
        self.model = models.load_model(filename)
