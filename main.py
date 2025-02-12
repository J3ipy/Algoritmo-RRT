import numpy as np
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RRTVisualizer:
    def __init__(self, image_path, start, goal=None):
        self.img = Image.open(image_path).convert('L')
        self.img_array = np.array(self.img)
        self.width, self.height = self.img.size
        
        # Verifica posições iniciais
        if self.img_array[start[1], start[0]] < 128:
            raise ValueError("Start position inválida!")
        if goal and self.img_array[goal[1], goal[0]] < 128:
            raise ValueError("Goal position inválida!")
        
        # Configurações do plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(self.img_array, cmap='gray', origin='upper')
        self.ax.set_title("RRT - Planejamento de Caminho")
        
        # Elementos gráficos aprimorados
        self.start_marker = self.ax.scatter(start[0], start[1], c='lime', s=300, 
                                          edgecolors='black', marker='o', zorder=3)
        self.goal_marker = self.ax.scatter([], [], c='cyan', s=300, edgecolors='black', 
                                         marker='*', zorder=3) if goal else None
        self.tree_lines = []
        self.path_line, = self.ax.plot([], [], 'y-', linewidth=5, zorder=2)
        self.current_cost = 0.0
        self.info_text = self.ax.text(0.02, 0.97, '', transform=self.ax.transAxes, 
                                    fontsize=12, color='black',
                                    bbox=dict(facecolor='white', alpha=0.9))
        
        # Parâmetros RRT
        self.start = start
        self.goal = goal
        self.tree = [start]
        self.parents = {0: -1}
        self.step_size = 25
        self.goal_threshold = 15
        self.max_iterations = 2000
        self.current_iteration = 0
        self.goal_reached = False
        self.final_path = []
        self.path_cost = 0.0

    def is_collision(self, start, end):
        line_points = self.bresenham_line(start, end)
        for (x, y) in line_points:
            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                return True
            if self.img_array[y, x] < 128:
                return True
        return False

    @staticmethod
    def bresenham_line(start, end):
        x0, y0 = int(round(start[0])), int(round(start[1]))
        x1, y1 = int(round(end[0])), int(round(end[1]))
        
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def find_nearest(self, point):
        min_dist = float('inf')
        nearest = None
        for node in self.tree:
            dist = math.hypot(node[0]-point[0], node[1]-point[1])
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest

    def steer(self, from_point, to_point):
        direction = np.array(to_point) - np.array(from_point)
        distance = np.linalg.norm(direction)
        if distance == 0:
            return from_point
        direction_unit = direction / distance
        return tuple((np.array(from_point) + direction_unit * min(distance, self.step_size)).astype(int))

    def update(self, frame):
        if self.goal_reached or self.current_iteration >= self.max_iterations:
            return [self.path_line, self.info_text]
        
        # Etapa de amostragem
        sample = self.goal if (self.goal and random.random() < 0.15) else (
            random.randint(0, self.width-1), 
            random.randint(0, self.height-1)
        )
        
        nearest = self.find_nearest(sample)
        new_point = self.steer(nearest, sample)
        
        if not self.is_collision(nearest, new_point) and self.img_array[new_point[1], new_point[0]] >= 128:
            self.tree.append(new_point)
            self.parents[len(self.tree)-1] = self.tree.index(nearest)
            
            # Desenha nova conexão
            line, = self.ax.plot([nearest[0], new_point[0]], [nearest[1], new_point[1]], 
                                color='#FF4500', linewidth=2, alpha=0.7, zorder=1)
            self.tree_lines.append(line)
            
            # Verifica goal
            if self.goal and math.hypot(new_point[0]-self.goal[0], new_point[1]-self.goal[1]) < self.goal_threshold:
                self.goal_reached = True
                self.extract_final_path(new_point)
                self.animate_final_path()
                
        self.current_iteration += 1
        self.update_info()
        return [self.path_line, self.info_text] + self.tree_lines

    def extract_final_path(self, last_node):
        current = self.tree.index(last_node)
        while current != 0:
            self.final_path.append(self.tree[current])
            current = self.parents[current]
        self.final_path.append(self.start)
        self.final_path.reverse()
        
        # Calcula custo total
        self.path_cost = sum(math.hypot(self.final_path[i][0]-self.final_path[i-1][0],
                                      self.final_path[i][1]-self.final_path[i-1][1])
                           for i in range(1, len(self.final_path)))

    def animate_final_path(self):
        # Prepara animação do caminho
        for i in range(1, len(self.final_path)+1):
            segment = self.final_path[:i]
            x_vals = [p[0] for p in segment]
            y_vals = [p[1] for p in segment]
            
            # Atualiza linha do caminho
            self.path_line.set_data(x_vals, y_vals)
            
            # Atualiza custo parcial
            partial_cost = sum(math.hypot(segment[j][0]-segment[j-1][0],
                                        segment[j][1]-segment[j-1][1])
                             for j in range(1, len(segment)))
            
            # Atualiza texto
            self.info_text.set_text(
                f"Iterações: {self.current_iteration}\n"
                f"Custo total: {self.path_cost:.2f}\n"
                f"Custo atual: {partial_cost:.2f}"
            )
            
            # Destaca ponto atual
            if i < len(self.final_path):
                self.ax.plot(segment[-1][0], segment[-1][1], 'yo', 
                           markersize=10, alpha=0.8, zorder=2)
            
            plt.pause(0.05)
        
        # Texto final
        self.info_text.set_text(
            f"* Caminho encontrado! *\n"
            f"Iterações: {self.current_iteration}\n"
            f"Custo total: {self.path_cost:.2f}"
        )
        plt.draw()

    def update_info(self):
        text = f"Iterações: {self.current_iteration}"
        if self.goal_reached:
            text += f"\nCusto total: {self.path_cost:.2f}"
        self.info_text.set_text(text)

    def run(self):
        # Atualiza marcador do goal
        if self.goal:
            self.goal_marker = self.ax.scatter(self.goal[0], self.goal[1], c='cyan', s=300,
                                             edgecolors='black', marker='*', zorder=3)
        
        plt.ion()
        plt.show()
        while not self.goal_reached and self.current_iteration < self.max_iterations:
            self.update(None)
            plt.pause(0.001)
        plt.ioff()
        if not self.goal_reached:
            print("Goal não alcançado!")
        plt.show()

# Exemplo de uso
if __name__ == "__main__":
    # Configure conforme sua imagem 578x457
    start = (50, 50)
    goal = (520, 380)  # Ajuste as coordenadas
    
    visualizer = RRTVisualizer(
        image_path='mapa.png',
        start=start,
        goal=goal
    )
    
    # Parâmetros ajustáveis
    visualizer.step_size = 20
    visualizer.max_iterations = 2000
    visualizer.goal_threshold = 15
    
    visualizer.run()