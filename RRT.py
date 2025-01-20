import matplotlib.pyplot as plt
import random
import math

# Parâmetros do algoritmo
NUM_NODES = 500
STEP_SIZE = 0.5
OBSTACLES = [((3, 3), (2, 2)), ((6, 6), (1.5, 1.5)), ((1, 7), (2, 1)), ((7,8), (1,4))]

# Classe para representar um nó
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

# Gera um ponto aleatório nos limites
def generate_random_point(xlim, ylim):
    return random.uniform(*xlim), random.uniform(*ylim)

# Calcula a distância euclidiana
def distance(node1, node2):
    return math.hypot(node1.x - node2.x, node1.y - node2.y)

# Encontra o nó mais próximo
def nearest_node(tree, point):
    return min(tree, key=lambda node: distance(node, Node(*point)))

# Cria um novo nó em direção ao ponto aleatório
def new_node(nearest, point, step_size):
    angle = math.atan2(point[1] - nearest.y, point[0] - nearest.x)
    x = nearest.x + step_size * math.cos(angle)
    y = nearest.y + step_size * math.sin(angle)
    return Node(x, y, parent=nearest)

# Verifica se um ponto está dentro de um obstáculo
def is_in_obstacle(point):
    px, py = point
    for (ox, oy), (w, h) in OBSTACLES:
        if ox <= px <= ox + w and oy <= py <= oy + h:
            return True
    return False

# Construção da árvore RRT
def build_rrt(start, goal, xlim, ylim):
    tree = [Node(*start)]
    plt.figure(figsize=(8, 8))
    plt.plot(*start, 'go', markersize=12, label="Início")
    plt.plot(*goal, 'ro', markersize=12, label="Objetivo")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.title("RRT em Progresso")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Desenha os obstáculos
    for (ox, oy), (w, h) in OBSTACLES:
        plt.gca().add_patch(plt.Rectangle((ox, oy), w, h, color='gray', alpha=0.5))

    for iteration in range(1, NUM_NODES + 1):
        random_point = generate_random_point(xlim, ylim)
        if is_in_obstacle(random_point):
            continue

        nearest = nearest_node(tree, random_point)
        new = new_node(nearest, random_point, STEP_SIZE)

        if is_in_obstacle((new.x, new.y)):
            continue

        tree.append(new)

        # Visualiza o progresso
        plt.plot([nearest.x, new.x], [nearest.y, new.y], color='black')  # Nós visitados em preto
        plt.plot(new.x, new.y, 'go', markersize=2)  # Nós novos em verde
        plt.pause(0.02)

        path, cost = get_path(new)
        print(f"Iteração: {iteration}, Custo atual: {cost:.2f}")  # Imprime no terminal


        if distance(new, Node(*goal)) < STEP_SIZE:
            goal_node = Node(*goal, parent=new)
            tree.append(goal_node)
            return tree, goal_node, iteration

    return tree, None, NUM_NODES

# Extrai o caminho do nó objetivo até o início
def get_path(node):
    path = []
    cost = 0
    while node:
        if node.parent:
            cost += distance(node, node.parent)
        path.append((node.x, node.y))
        node = node.parent
    return path[::-1], cost

# Parâmetros iniciais
start = (0, 0)
goal = (9, 9)
xlim = (0, 10)
ylim = (0, 10)

# Executa o RRT e obtém o caminho
tree, goal_node, iterations = build_rrt(start, goal, xlim, ylim)
if goal_node:
    path, cost = get_path(goal_node)
else:
    path, cost = [], float('inf')

# Visualização do caminho final passo a passo
if goal_node:
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        plt.plot([x1, x2], [y1, y2], 'r-', linewidth=2)  # Linha vermelha para o caminho
        plt.plot(x1, y1, 'bo', markersize=8)  # Ponto atual destacado
        plt.pause(0.5)

    plt.plot(path[-1][0], path[-1][1], 'bo', markersize=8, label="Caminho encontrado")  # Último ponto
    plt.legend()
    plt.title(f"RRT Concluído - Iterações: {iterations}, Custo do caminho encontrado: {cost:.2f}")
    plt.show()
else:
    plt.title(f"RRT Falhou - Iterações: {iterations}")
    plt.show()
