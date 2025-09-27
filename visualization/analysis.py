
""" Назначение: Визуализация средних показателей """


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_agent(env, agent, episodes=10):
    """
    Анализирует эффективность агента.
    """

    total_rewards = []
    total_costs = []
    total_power_usage = []

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        total_cost = 0
        power_usage_per_day = []

        for time in range(len(env.energy_cost)):
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            power_usage = np.dot(action[:-1], env.devices)
            cost = power_usage * env.energy_cost[env.time]

            power_usage_per_day.append(power_usage)
            total_reward += reward
            total_cost += cost

            state = next_state
            if done:
                break

        total_rewards.append(total_reward)
        total_costs.append(total_cost)
        total_power_usage.append(power_usage_per_day)

    avg_reward = np.mean(total_rewards)
    avg_cost = np.mean(total_costs)
    avg_power_usage = np.mean(total_power_usage, axis=0)

    print(f"Средняя награда за {episodes} эпизодов: {avg_reward:.2f}")
    print(
        f"Средняя стоимость энергии за {episodes} эпизодов: {avg_cost:.2f} руб"
    )

    # Визуализация среднего энергопотребления
    plt.figure(figsize=(12, 6))
    plt.plot(avg_power_usage, label="Среднее энергопотребление", marker='o')
    plt.axhline(env.max_power, color='r', linestyle='--',
                label="Максимальная мощность")
    plt.xlabel('Часы суток')
    plt.ylabel('Энергопотребление (Вт)')
    plt.title('Среднее энергопотребление по времени суток')
    plt.legend()
    plt.grid()
    plt.show()

    # Диаграмма рассеяния общих расходов
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=list(range(1, episodes+1)),
                    y=total_costs, color='darkorange')
    plt.axhline(avg_cost, color='k', linestyle=':', label='Средняя стоимость')
    plt.xlabel('Эпизод')
    plt.ylabel('Общие расходы (руб.)')
    plt.title('Общая стоимость электроэнергии по эпизодам')
    plt.legend()
    plt.grid()
    plt.show()

    # Итоговые графики производительности
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    axes[0].plot(list(range(1, episodes+1)),
                 total_rewards, color='forestgreen')
    axes[0].set_title('Награда по эпизодам')
    axes[0].set_xlabel('Эпизод')
    axes[0].set_ylabel('Суммарная награда')
    axes[0].grid(True)

    axes[1].plot(avg_power_usage, color='royalblue', marker='.')
    axes[1].axhline(env.max_power, color='crimson',
                    linestyle='--', label='Предельная мощность')
    axes[1].legend(loc='upper right')
    axes[1].set_title('Среднее энергопотребление по времени')
    axes[1].set_xlabel('Часы')
    axes[1].set_ylabel('Энергопотребление (Вт)')
    axes[1].grid(True)

    plt.suptitle('Анализ работы агента', fontsize=16)
    plt.show()

    return total_rewards, total_costs, total_power_usage
