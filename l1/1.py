import sys
import time
from copy import copy, deepcopy
import random
from math import sqrt
import numpy
import tsplib95
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod


# wypisywanie cyklu
def print_result(result):
    for i in range(len(result)):
        print(result[i] + 1, "-> ", end="")
    print(result[0] + 1)


# odwracanie czesci listy
def invert(vector, i, j):
    vector[i:j + 1] = vector[i:j + 1][::-1]
    return vector


# klasa abstrakcyjna grafu
class Graph(ABC):

    # macierz sasiedztwa
    def __init__(self):
        self.matrix = []

    # ladowanie grafu z pliku
    @abstractmethod
    def load(self, filename):
        pass

    # generowanie losowego grafu
    @abstractmethod
    def random(self, size):
        pass

    # wypisywanie grafu
    def print_instance(self):
        for i in range(len(self.matrix)):
            print(self.matrix[i])

    # funkcja celu
    def f(self, cycle):
        sum = 0
        for c in range(len(cycle) - 1):
            try:
                sum += self.matrix[cycle[c]][cycle[c + 1]]
            except IndexError:
                sum += self.matrix[cycle[c + 1]][cycle[c]]
        try:
            sum += self.matrix[cycle[len(cycle) - 1]][cycle[0]]
        except IndexError:
            sum += self.matrix[cycle[0]][cycle[len(cycle) - 1]]
        return sum

    # blad wzgledny
    def PRD(self, best_result, result):
        if type(best_result) == int:
            f_res = self.f(result)
            f_best = best_result
            return ((f_res - f_best) / f_best) * 100
        elif type(best_result) == list:
            f_res = self.f(result)
            f_best = self.f(best_result)
            return ((f_res - f_best) / f_best) * 100

    # wybieranie zawsze najblizszego sasaiada
    def nearest_neighbour(self, position=-1):
        # albo dostajemy albo losujemy pozycje startowa
        if position == -1:
            position = random.randint(0, len(self.matrix) - 1)
        result = [position]
        # odpowiednie kopiowanie tablicy, bedziemy z niej "usuwac" odwiedzone wierzcholki
        if isinstance(self.matrix, Full):
            actual = [row[:] for row in self.matrix]
        else:
            actual = [[0 for i in range(len(self.matrix))] for j in range(len(self.matrix))]
            for i in range(len(self.matrix)):
                for j in range(i):
                    value = self.matrix[i][j]
                    actual[i][j] = value
                    actual[j][i] = value
        # wybieramy najblizszy z jeszcze nieodwiedzonych wierzcholkow
        while len(result) != len(actual[0]):
            for i in range(len(actual)):
                actual[i][position] = sys.maxsize
            position = actual[position].index(min(actual[position]))
            result.append(position)

        return result

    # wybieranie najlepszego z k losowych rozwiazan
    def krandom(self, k):
        best = [i for i in range(len(self.matrix))]
        cost = self.f(best)
        for _ in range(k):
            startprim = [i for i in range(len(self.matrix))]
            random.shuffle(startprim)
            currentcost = self.f(startprim)
            if currentcost < cost:
                cost = currentcost
                best = startprim
        return best

    def twoOPT(self, sizeN):
        # sizeN to jak wielkie ma być sąsiedztwo

        # rozwiązanie początkowe
        start = [i for i in range(len(self.matrix))]
        random.shuffle(start)

        cycle = copy(start)
        cycle.extend(start[0:1])
        startcost = self.f(cycle)

        while True:
            # tworzenie otoczenia
            N = list()
            for i in range(sizeN):
                invertstart = random.randint(0, len(self.matrix) - 1)
                invertend = random.randint(0, len(self.matrix) - 1)

                if invertstart > invertend:
                    invertstart, invertend = invertend, invertstart

                # invert
                st = copy(start)
                pi = invert(st, invertstart, invertend)

                if invertstart != invertend:
                    N.append(pi)

            # Czy wygenerowane sąsieddzctwo posiada lepsze PI
            isBetter = False
            for n in N:
                piprim = copy(n)
                piprim.extend(n[0:1])
                currentcost = self.f(piprim)

                if currentcost < startcost:
                    start = piprim[0:len(piprim) - 1]
                    startcost = currentcost
                    isBetter = True

            if not isBetter:
                return start


class Full(Graph):
    def load(self, filename):
        problem = tsplib95.load(filename)
        nodeList = list(problem.get_nodes())

        self.matrix = [[None for i in range(len(nodeList))] for j in range(len(nodeList))]

        for i in nodeList:
            for j in nodeList:
                self.matrix[i][j] = problem.get_weight(i, j)

    def random(self, size):
        self.matrix = [[None for i in range(size)] for j in range(size)]
        for i in range(size):
            for j in range(size):
                self.matrix[i][j] = random.randint(0, 1000)


class Lower(Graph):
    def load(self, filename):
        problem = tsplib95.load(filename)
        nodeList = list(problem.get_nodes())
        self.matrix = [[0 for i in range(j + 1)] for j in range(len(nodeList))]
        for i in nodeList:
            for j in range(1, i):
                self.matrix[i - 1][j - 1] = problem.get_weight(i, j)

    def random(self, size):
        self.matrix = [[0 for i in range(j + 1)] for j in range(size)]
        for i in range(size):
            for j in range(i):
                self.matrix[i][j] = random.randint(0, 1000)


class Euc2D(Graph):
    def __init__(self):
        Graph.__init__(self)
        # wspolrzedne
        self.y = []
        self.x = []

    def load(self, filename):
        problem = tsplib95.load(filename)
        nodeList = list(problem.get_nodes())
        edgesList = list(problem.get_edges())

        self.matrix = [[None for i in range(j + 1)] for j in range(len(nodeList))]

        for i in range(1, len(nodeList) + 1):
            self.x.append(problem.node_coords[i][0])
            self.y.append(problem.node_coords[i][1])

        for i in range(len(edgesList)):
            edge = edgesList[i]
            try:
                self.matrix[edge[0] - 1][edge[1] - 1] = problem.get_weight(*edge)
            except IndexError:
                pass

    def random(self, size):
        self.matrix = [[0 for i in range(j + 1)] for j in range(size)]
        for i in range(size):
            self.x.append(random.randint(0, 1000))
            self.y.append(random.randint(0, 1000))

        for i in range(size):
            for j in range(i):
                x1, y1 = (self.x[i], self.y[i])
                x2, y2 = (self.x[j], self.y[j])
                self.matrix[i][j] = int(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))  # część rzeczywista liczby zespolonej

    # graficzne przedstaweienie wyniku
    def result_graphic(self, result):
        plt.scatter(self.x, self.y)
        for i in range(len(self.x)):
            plt.text(self.x[i], self.y[i], i + 1)
        plt.plot([self.x[result[i]] for i in range(len(result))], [self.y[result[i]] for i in range(len(result))],
                 color="red")
        plt.plot([self.x[result[len(result) - 1]], self.x[result[0]]],
                 [self.y[result[len(result) - 1]], self.y[result[0]]],
                 color="red")
        plt.show()


def timeStats(graph):
    sizes = [10, 50, 100, 200, 300]
    scores_krandom=[]
    scores_nearest = []
    scores_opt = []
    for size in sizes:
        score1=0
        score2=0
        score3 = 0
        for i in range(10):
            graph.random(size)
            start = time.time()
            graph.krandom(10)
            end = time.time()
            score1 += (end - start)
            start = time.time()
            graph.nearest_neighbour()
            end = time.time()
            score2 += (end - start)
            start = time.time()
            graph.twoOPT(int(size/4))
            end = time.time()
            score3 += (end - start)
        scores_krandom.append(score1 / 10)
        scores_nearest.append(score2 / 10)
        scores_opt.append(score3/10)
    plt.plot(sizes,scores_krandom,
                 color="red",marker="o",label="k-random")
    plt.plot(sizes,scores_nearest,
             color="green",marker="o",label="nearest neighbour")
    plt.plot(sizes,scores_opt,
             color="blue",marker="o",label="2-OPT")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.title("Time complexity of heuristic algorithms")
    plt.xlabel("Size")
    plt.ylabel("Time")
    plt.show()

full=Full()
timeStats(full)


