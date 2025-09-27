
""" Назначение: Файл реализации среды и методов её настройки """


import numpy as np


class SmartHomeEnvironmentBattery:
    def __init__(self, devices, max_power, energy_cost, battery_capacity):
        """
        Конструктор класса, задаёт параметры среды.

        :param devices: список энергопотребления устройств в ваттах.
        :param max_power: максимальная доступная мощность в ваттах.
        :param energy_cost: стоимость электроэнергии в рублях за киловатт-час.
        :param battery_capacity: емкость аккумулятора в Втч.
        """

        # Потребляемая мощность устройств
        self.devices = devices
        # Верхний порог потребления энергии
        self.max_power = max_power
        # Стоимость энергии по времени
        self.energy_cost = energy_cost
        # Количество устройств
        self.num_devices = len(devices)
        # Начальное состояние (все устройства выключены)
        self.state = np.zeros(self.num_devices + 1)
        # Начальное время (начало дня)
        self.time = 0
        # Емкость аккумулятора
        self.battery_capacity = battery_capacity
        # Текущий уровень заряда аккумулятора
        self.battery_level = 0

    def step(self, actions):
        """
        Функция совершает шаг в симуляции с учетом управления аккумулятором

        actions: список действий (0 или 1) для каждого устройства
        return: новое состояние, награда и признак окончания периода
        """

        # Первые элементы списка — включение устройств
        device_actions = actions[:-1]
        # Последний элемент управляет аккумулятором
        # (-1 разряжать, 0 ничего, 1 заряжать)
        battery_action = actions[-1]

        # Заряжаем или разряжаем аккумулятор
        charge_amount = 0
        if battery_action == 1:  # Заряжаем аккумулятор
            available_charge = self.max_power - \
                np.dot(device_actions, self.devices)
            charge_amount = min(
                available_charge, self.battery_capacity - self.battery_level)
        elif battery_action == -1:  # Разряжаем аккумулятор
            discharge_amount = self.battery_level
            # Отрицательное значение означает разряд
            charge_amount = -discharge_amount

        # Изменяем уровень заряда аккумулятора
        self.battery_level += charge_amount

        # Определяем потребление энергии с учётом аккумулятора
        power_usage = np.dot(device_actions, self.devices) - charge_amount

        # Проверка превышения максимального порога потребляемой мощности
        penalty = 0
        if power_usage > self.max_power:
            penalty = -(power_usage - self.max_power) * 0.1

        # Рассчитываем стоимость потребления энергии
        cost = power_usage * self.energy_cost[self.time]
        reward = -cost + penalty  # Награда учитывает стоимость и штрафы

        # Обновление временного слота
        self.time = (self.time + 1) % len(self.energy_cost)

        # Новое состояние включает информацию о заряде аккумулятора
        new_state = np.concatenate((device_actions, [self.battery_level]))

        # Возвращаем новую ситуацию, награду и признак завершения дня
        return new_state, reward, self.time == 0

    def reset(self):
        """
        Сбрасывает состояние среды обратно в начало.
        """

        # Сброс состояния
        self.state = np.zeros(self.num_devices + 1)
        # Возврат ко времени старта
        self.time = 0
        # Возвращаем стартовое состояние
        return self.state

    def update_environment(
        self,
        devices=None,
        max_power=None,
        energy_cost=None
    ):
        """
        Метод для обновления параметров среды.

        :param devices: новый список энергопотребления устройств.
        :param max_power: новое значение максимальной мощности.
        :param energy_cost: новые расценки на электроэнергию.
        """

        if devices is not None:
            # Обновляем список устройств
            self.devices = devices
            # Обновляем количество устройств
            self.num_devices = len(devices)
            # Переинициализируем состояние
            self.state = np.zeros(self.num_devices + 1)
        if max_power is not None:
            # Обновляем максимальный лимит мощности
            self.max_power = max_power
        if energy_cost is not None:
            # Обновляем стоимость энергии
            self.energy_cost = energy_cost
