
from operator import le
import re
import sys
import os
import psutil
import time
from copy import copy, deepcopy
import random
from math import sqrt
import tsplib95
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from Charts import Charts
from heapq import *
import queue
from collections import deque


# wypisywanie cyklu
def print_result(result):
    for i in range(len(result)):
        print(result[i] + 1, "-> ", end="")
    print(result[0] + 1)


# odwracanie czesci listy
def invert(vector, i, j):
    vector[i:j + 1] = vector[i:j + 1][::-1]
    return vector


# zamiana dwoch elementow
def swap(vector, i, j):
    vector[i], vector[j] = vector[j], vector[i]
    return vector


# wstawienie i-tego elemntu na miejsce j-te
def insert(vector, i, j):
    new = vector[0:i] + vector[i + 1:]
    new.insert(j, vector[i])
    return new


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
        # print("Cycle size: ",sum)
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
            actual = [[0 for _ in range(len(self.matrix))] for _ in range(len(self.matrix))]
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
        random.shuffle(best)
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

    # argumenty to rozwiazanie poczatkowe, lub funkcja ktora je generuje wraz z argumentem,
    # funkcja ktora generuje sasiedztwo wraz z jego rozmiarem,
    # rozmiar listy tabu
    # warunek stopu
    def tabuSearch(self, start, arg, neighbourhood, size=100, tabuSize=10, stop=10000, decision="1"):
        if callable(start):
            solution = start(arg)
        elif type(start) == list:
            solution = start
        else:
            solution = [i for i in range(len(self.matrix))]
            random.shuffle(solution)
        best=self.f(solution) # wyznaczenie rozwiazania poczatkowego 

        iteration = 0
        iterationNoChange = 0
        tabooList = MyStruct(tabuSize)

       # while iterationNoChange<stop:
        while iteration < stop:
            iteration+=1
            N = []
            for _ in range(size):
                while True:
                    s = random.randint(0, len(self.matrix) - 1)
                    e = random.randint(0, len(self.matrix) - 1)
                    if s!=e:
                        break
                if s > e:
                    s, e = e, s
                N.append((s, e))

            NwithoutTaboo = copy(N)
            for t in tabooList.queueToArray():
                while t in NwithoutTaboo:
                    NwithoutTaboo.remove(t)

            if len(NwithoutTaboo) == 0:
                print("wszedl")
                match decision:
                    case "1": 
                        return solution
                    case "2":
                        NwithoutTabooPrim = copy(N)
                        tabooListPrim = copy(tabooList.queueToArray())
                        print(tabooListPrim, NwithoutTabooPrim)

                        while True:
                            tabooListPrim.pop()
                            print(tabooListPrim, NwithoutTabooPrim)
                            for t in tabooListPrim:
                                while t in NwithoutTabooPrim:
                                    NwithoutTabooPrim.remove(t)
                            if len(NwithoutTabooPrim) != 0:
                                break
                            if len(tabooListPrim) == 0:
                                break
                        NwithoutTaboo = NwithoutTabooPrim
                    case "3":
                        self.tabuSearch(start, arg, neighbourhood, size, tabuSize, stop, decision)
                    case "4":
                        pass
                    case "5":
                        pass
                    case _:
                        return solution
                return 

            pi = min(NwithoutTaboo, key=lambda t: self.f(neighbourhood(copy(solution), t[0], t[1])))
            tabooList.push((pi[0], pi[1]))
    
            neighbour = neighbourhood(solution, pi[0], pi[1])
            new=self.f(neighbour)
            if new<best:
                solution=neighbour
                best=new
            else:
                iterationNoChange += 1
        return solution


class MyStruct():
    def __init__(self, size):
        self.q = queue.Queue()
        self.size = size
        self.ln = 0;
    
    def push(self, element):
        if (self.ln < self.size):
            self.q.put(element)
            self.ln += 1
        else:
            self.q.get()
            self.q.put(element)

    def pop(self):
        self.ln -= 1
        return self.q.get()

    def queueToArray(self):
        l = list(self.q.queue)
        lfinal = list()
        for i in l:
            lfinal.append((i[0], i[1]))
        return lfinal
        # return list(self.q.queue)

# m = MyStruct(5)
# m.push((1, 0))
# m.push((2, 0))
# m.push((3, 0))
# m.push((4, 0))
# m.push((5, 0))
# print(m.queueToArray())
# m.push((6, 0))
# print(m.queueToArray())

class Full(Graph):
    def load(self, filename):
        problem = tsplib95.load(filename)
        nodeList = list(problem.get_nodes())

        self.matrix = [[None for i in range(len(nodeList))] for j in range(len(nodeList))]

        for i in nodeList:
            for j in nodeList:
                self.matrix[i - 1][j - 1] = problem.get_weight(i, j)

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


# def measurememory(obj, filename, k, type):
#     matrix = obj
#     matrix.load(filename)
#     process = psutil.Process(os.getpid())
#     if type == "twoOPT":
#         mem_before = process.memory_info().rss
#         matrix.twoOPT(2 ** k)
#         mem_after = process.memory_info().rss
#         print(obj.__class__.__name__, k, mem_after - mem_before)
#     elif type == "krandom":
#         mem_before = process.memory_info().rss
#         matrix.krandom(2 ** k)
#         mem_after = process.memory_info().rss
#         print(obj.__class__.__name__, k, mem_after - mem_before)
#     elif type == "nearest":
#         mem_before = process.memory_info().rss
#         matrix.nearest_neighbour()
#         mem_after = process.memory_info().rss
#         print(obj.__class__.__name__, k, mem_after - mem_before)
#
#     del matrix, mem_before, mem_after
#
#
# def measurePSD(obj, filename, k=14, freftsplib=-1):
#     # how to run? i. e. measurePSD(Full(), 'berlin52.tsp')
#     # or measurePSD(Full(), 'berlin52.tsp', freftsplib=7542), but in this case we have to know which is the best optional solution
#     # http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html
#     matrix = obj
#     if type(filename) == String:
#         matrix.load(filename)
#     else:
#         matrix.random(filename)
#
#     c1 = matrix.krandom(2 ** k)
#     f1 = matrix.f(c1)
#     c2 = matrix.twoOPT(2 ** k)
#     f2 = matrix.f(c2)
#     c3 = matrix.nearest_neighbour()
#     f3 = matrix.f(c3)
#
#     pdrArray = list()
#     pdrArray.append(f1)
#     pdrArray.append(f2)
#     pdrArray.append(f3)
#
#     fref = freftsplib
#     if fref == -1:
#         fref = min(pdrArray)
#         print(fref)
#
#     frefArray = list()
#     for i in range(len(pdrArray)):
#         frefArray.append((pdrArray[i] - fref) / fref * 100)
#
#     # freeArray[0] - krandom result
#     # freeArray[1] - 2OPT result
#     # freeArray[2] - nearest neighbour
#     return frefArray
#
#
def timeStats(graph):
    sizes = [10, 50, 100, 200, 300]
    scores_krandom = []
    scores_nearest = []
    scores_opt = []
    for size in sizes:
        score1 = 0
        score2 = 0
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
            graph.twoOPT(10)
            end = time.time()
            score3 += (end - start)
        scores_krandom.append(score1 / 10)
        scores_nearest.append(score2 / 10)
        scores_opt.append(score3 / 10)
    chart = Charts("Time complexity of heuristic algorithms", "Size", "Time")
    chart.load(sizes, scores_krandom, "red", "k-random")
    chart.load(sizes, scores_nearest, "green", "nearest-neighbour")
    chart.load(sizes, scores_opt, "blue", "2-opt")
    chart.plot()


def memoryStats(graph):
    process = psutil.Process(os.getpid())
    sizes = [10, 50, 100, 200, 300]
    scores_krandom = []
    scores_nearest = []
    scores_opt = []
    for size in sizes:
        score1 = 0
        score2 = 0
        score3 = 0
        for i in range(10):
            graph.random(size)
            start = process.memory_info().rss
            graph.krandom(10)
            end = process.memory_info().rss
            score1 += (end - start)
            start = process.memory_info().rss
            graph.nearest_neighbour()
            end = process.memory_info().rss
            score2 += (end - start)
            start = process.memory_info().rss
            graph.twoOPT(10)
            end = process.memory_info().rss
            score3 += (end - start)
        scores_krandom.append(score1 / 10)
        scores_nearest.append(score2 / 10)
        scores_opt.append(score3 / 10)
    chart = Charts("Memory complexity of heuristic algorithms", "Memory", "Time")
    chart.load(sizes, scores_krandom, "red", "k-random")
    chart.load(sizes, scores_nearest, "green", "nearest-neighbour")
    chart.load(sizes, scores_opt, "blue", "2-opt")
    chart.plot()


def PRDStats(graph):
    sizes = [10, 100, 500, 1000, 5000, 10000]
    scores_krandom = []
    scores_nearest = []
    scores_opt = []
    for size in sizes:
        score1 = 0
        score2 = 0
        score3 = 0
        for i in range(10):
            graph.random(size)
            result1 = graph.krandom(size)
            result2 = graph.nearest_neighbour()
            result3 = graph.twoOPT(size)
            mini = min([graph.f(result1), graph.f(result2), graph.f(result3)])
            score1 += graph.PRD(mini, result1)
            score2 += graph.PRD(mini, result2)
            score3 += graph.PRD(mini, result3)
        scores_krandom.append(score1 / 10)
        scores_nearest.append(score2 / 10)
        scores_opt.append(score3 / 10)
    chart = Charts("PRD of heuristic algorithms", "PRD", "Time")
    chart.load(sizes, scores_krandom, "red", "k-random")
    chart.load(sizes, scores_nearest, "green", "nearest-neighbour")
    chart.load(sizes, scores_opt, "blue", "2-opt")
    chart.plot()

full = Full()
full.random(5)
# full.print_instance()
print_result(full.tabuSearch(full.twoOPT, 5, invert, 5, 20, 10, "2"))


# def test(instance,data,k,m,opt):
#     if type(data) is int:
#         instance.random(data)
#     else:
#         instance.load(data)
#     instance.print_instance()
#     for value in k:
#         print("K-RANDOM ",value)
#         result= instance.krandom(value)
#         print_result(result)
#         if type(instance) is Euc2D:
#             instance.result_graphic(result)
#         print("Cycle size: ", instance.f(result))
#         print("PRD: ", instance.PRD(opt, result))
#     print("NEAREST NEIGHBOUR")
#     result = instance.nearest_neighbour()
#     print_result(result)
#     if type(instance) is Euc2D:
#         instance.result_graphic(result)
#     print("Cycle size: ", instance.f(result))
#     print("PRD: ", instance.PRD(opt, result))
#     for value in m:
#         print("2-OPT ", value)
#         result = instance.twoOPT(value)
#         print_result(result)
#         if type(instance) is Euc2D:
#             instance.result_graphic(result)
#         print("Cycle size: ", instance.f(result))
#         print("PRD: ", instance.PRD(opt, result))


# test(Full(),"ftv47.atsp",[10,1000,100000],[10000,100],1776)


# full=Full()
# full.load("ftv47.atsp")
# full.print_instance()
# fullRes=full.krandom(10)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(1776,fullRes))
# fullRes=full.krandom(1000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(1776,fullRes))
# fullRes=full.krandom(100000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(1776,fullRes))
# fullRes=full.nearest_neighbour()
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(1776,fullRes))
# fullRes=full.twoOPT(10000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(1776,fullRes))


# full=Euc2D()
# full.load("berlin52.tsp")
# full.print_instance()
# fullRes=full.krandom(10)
# full.result_graphic(fullRes)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(7542,fullRes))
# fullRes=full.krandom(1000)
# full.result_graphic(fullRes)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(7542,fullRes))
# fullRes=full.krandom(100000)
# full.result_graphic(fullRes)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(7542,fullRes))
# fullRes=full.nearest_neighbour()
# full.result_graphic(fullRes)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(7542,fullRes))
# fullRes=full.twoOPT(1000)
# full.result_graphic(fullRes)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(7542,fullRes))

# full=Lower()
# full.load("gr120.tsp")
# full.print_instance()
# fullRes=full.krandom(10)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(6942,fullRes))
# fullRes=full.krandom(1000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(6942,fullRes))
# fullRes=full.krandom(100000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(6942,fullRes))
# fullRes=full.nearest_neighbour()
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(6942,fullRes))
# fullRes=full.twoOPT(1000)
# print_result(fullRes)
# print("Cycle size: ",full.f(fullRes))
# print("PRD: ",full.PRD(6942,fullRes))
