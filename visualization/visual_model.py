
""" Назначение: Анализ производительности агента """


import matplotlib.pyplot as plt
import networkx as nx


def visualize_state(state):
    """
    Функция визуализации текущего состояния устройств.
    :param state: состояние устройств (включено/выключено).
    """

    plt.bar(range(len(state)), state,
            # Синяя полоска — включено, красная — выключено
            color=['blue' if s == 1 else 'red' for s in state])
    plt.xlabel('Устройства')
    plt.ylabel('Состояние (1 — включено, 0 — выключено)')
    plt.title('Текущее состояние устройств')
    plt.xticks(range(len(state)), [
               f'Устройство {i+1}' for i in range(len(state))])
    # Диапазон шкалы по y
    plt.ylim(-0.5, 1.5)
    plt.show()


# Визуализация архитектуры
def visualize_network_architecture(input_size, hidden_sizes, output_size):
    """
    Функция визуализации архитектуры нейронной сети.
    :param input_size: количество входных нейронов.
    :param hidden_sizes: размеры скрытых слоёв.
    :param output_size: количество выходных нейронов.
    """

    # Создаем ориентированный граф
    G = nx.DiGraph()

    # Входной слой
    for i in range(input_size):
        # Узлы входа помечаются слоем 0
        G.add_node(f"Input_{i+1}", layer=0)

    # Скрытые слои
    for l, size in enumerate(hidden_sizes):
        for i in range(size):
            # Узлы скрытого слоя помечаются уровнем слоя
            G.add_node(f"Hidden_{l+1}_{i+1}", layer=l+1)

    # Выходной слой
    for i in range(output_size):
        # Узлы выхода помечаются последним слоем
        G.add_node(f"Output_{i+1}", layer=len(hidden_sizes)+1)

    # Соединяем слои
    prev_layer_nodes = [f"Input_{i+1}" for i in range(input_size)]
    for l, size in enumerate(hidden_sizes):
        current_layer_nodes = [f"Hidden_{l+1}_{i+1}" for i in range(size)]
        for src in prev_layer_nodes:
            for dst in current_layer_nodes:
                # Связываем узлы предыдущего слоя с текущими
                G.add_edge(src, dst)
        prev_layer_nodes = current_layer_nodes

    # Связываем последний скрытый слой с выходом
    output_nodes = [f"Output_{i+1}" for i in range(output_size)]
    for src in prev_layer_nodes:
        for dst in output_nodes:
            G.add_edge(src, dst)

    # Рисуем граф
    # Позиционирование узлов по слоям
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(10, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=8,
        font_weight="bold",
        edge_color="gray"
    )
    plt.title("Архитектура нейронной сети")
    plt.show()


# Визуализация архитектуры
visualize_network_architecture(
    input_size=4,
    hidden_sizes=[24, 24],
    output_size=4
)
