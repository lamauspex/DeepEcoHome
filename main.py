
""" Назначение: Основной файл """


from model.train import run_training

from visualization.analysis import analyze_agent
from visualization.plots import visualize_dynamics

if __name__ == "__main__":
    run_training()

    # После основного цикла обучения визуализируем динамику обучения
    visualize_dynamics(rewards, epsilon_values)

    # Проводим анализ производительности агента
    rewards, costs, power_usage = analyze_agent(env, agent)
