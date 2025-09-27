
""" Назначение: Этот файл содержит основные циклы обучения агента """


import tqdm
from model.agent import DQNAgent
from model.env import SmartHomeEnvironmentBattery


if __name__ == "__main__":

    """ Настроим параметры среды и инициализируем среду и агента """
    # Мощность устройств
    devices = [100, 150, 200, 250]
    # Максимальная доступная мощность
    max_power = 500
    # Цены на энергию
    energy_cost = [1, 0.5, 0.2, 0.3, 0.4, 1.2, 1.5, 2]
    # Ёмкость аккумулятора Втч
    battery_capacity = 1000

    # Инстанцируем окружающую среду и агента
    env = SmartHomeEnvironmentBattery(
        devices, max_power, energy_cost, battery_capacity)
    state_size = env.num_devices
    action_size = state_size + 1
    agent = DQNAgent(state_size, action_size)

    # Заданные гиперпараметры
    episodes = 70         # Количество эпох обучения
    batch_size = 64       # Размер мини-батча
    rewards = []          # Хранение итоговых наград
    epsilon_values = []   # Изменение значения epsilon

    # Цикл обучения
    for e in tqdm(range(episodes)):
        state = env.reset()  # Начало нового эпизода
        total_reward = 0    # Итоговая награда за эпизод

        # Цикл шагов внутри одного дня
        for t in range(len(energy_cost)):
            action = agent.act(state)                     # Выбор действия
            next_state, reward, done = env.step(action)   # Выполнение шага

            # Логика запоминания и обучения
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # Процесс обучения (мини-батчи)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            # Завершаем эпизод, если достигли конца дня
            if done:
                break

        # Собираем статистику
        rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        # Сообщаем о прогрессе
        print(
            f"Эпоха: {e+1}/{episodes}",
            f"Награда: {total_reward:.2f}",
            f"Epsilon: {agent.epsilon:.2f}"
        )
