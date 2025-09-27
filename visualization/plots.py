
""" Назначение: Графики и диаграммы """


import matplotlib.pyplot as plt


def visualize_dynamics(rewards, epsilon_values):
    """Графическое отображение изменений в процессе обучения."""

    # Динамика суммарной награды
    plt.figure(figsize=(12, 5))
    plt.plot(rewards)
    plt.xlabel('Эпизод')
    plt.ylabel('Суммарная награда')
    plt.title('Динамика общей награды')
    plt.grid(True)
    plt.show()

    # Эволюция параметра epsilon
    plt.figure(figsize=(12, 5))
    plt.plot(epsilon_values, label='Epsilon decay')
    plt.axhline(y=1.0, color='grey', linestyle='--', label='Начальное epsilon')
    plt.axhline(y=0.01, color='black', linestyle='-',
                label='Минимальное epsilon')
    plt.xlabel('Эпизод')
    plt.ylabel('Значение epsilon')
    plt.title('Изменение вероятности случайного действия')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Визуализация нагрузки устройств и цен
    devices = [100, 150, 200, 250]
    energy_cost = [1, 0.5, 0.2, 0.3, 0.4, 1.2, 1.5, 2]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(devices)), devices)
    plt.xlabel('Устройства')
    plt.ylabel('Энергопотребление (Вт)')
    plt.title('Распределение мощностей устройств')

    plt.subplot(1, 2, 2)
    plt.plot(energy_cost)
    plt.xlabel('Время суток')
    plt.ylabel('Стоимость (руб./кВт·ч)')
    plt.title('Изменение тарифа электроэнергии')
    plt.tight_layout()
    plt.show()
