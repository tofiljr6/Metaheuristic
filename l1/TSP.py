from operator import le
import re
import sys
import os
from turtle import color
import psutil
import time
from copy import copy, deepcopy
import random
from math import sqrt
from pyparsing import col
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
        # wyznaczenie rozwiazania poczatkowego
        if callable(start):
            solution = start(arg)
        elif type(start) == list:
            solution = start
        else:
            solution = [i for i in range(len(self.matrix))]
            random.shuffle(solution)
        best = self.f(solution)

        # rozne warunki zakonczenie
        iteration = 0
        iterationNoChange = 0
        startTime = time.time()

        tabooList = MyStruct(tabuSize)
        aspirationList = MyStruct(tabuSize)

        # while time.time()-startTime<stop:
        # while iterationNoChange<stop:
        while iteration < stop:
            iteration += 1
            # generowanie sasiedztwa
            N = []
            for _ in range(size):
                while True:
                    s = random.randint(0, len(self.matrix) - 1)
                    e = random.randint(0, len(self.matrix) - 1)
                    if s != e:
                        break
                if s > e:
                    s, e = e, s
                N.append((s, e))

            # wybor najleszpego rozwiazania do kryterium asperacji
            bestTaboo = min(N, key=lambda t: self.f(neighbourhood(copy(solution), t[0], t[1])))

            # usuwania zabronionych sasiadow
            NwithoutTaboo = copy(N)
            for t in tabooList.queueToArray():
                while t in NwithoutTaboo:
                    NwithoutTaboo.remove(t)

            # co jesli wszystskie rozwiazania sa zabronione
            if len(NwithoutTaboo) == 0:
                match decision:
                    # konczymy algorytm
                    case "1":
                        return solution
                    # usuwamy zabrobione rozwiazania dopoki jakies bedzie legalne
                    case "2":
                        tabooListPrim = copy(tabooList.queueToArray())

                        while True:
                            NwithoutTabooPrim = copy(N)
                            tabooListPrim.pop()
                            for t in tabooListPrim:
                                while t in NwithoutTabooPrim:
                                    NwithoutTabooPrim.remove(t)
                            if len(NwithoutTabooPrim) != 0:
                                break
                        NwithoutTaboo = NwithoutTabooPrim
                        # tabooList = tabooListPrim # type - mish mash
                        tabooListStructPrim = MyStruct(tabuSize)
                        for element in tabooListPrim:
                            tabooListStructPrim.push(element)
                        tabooList = tabooListStructPrim

                    # zaczynamy algorytm z innym rozwiazaniem poczatkowym
                    case "3":
                        self.tabuSearch(start, arg, neighbourhood, size, tabuSize, stop, decision)
                    # robimy nawrot do aspirujacego rozwiazania
                    case "4":
                        asp = aspirationList.pop()
                        solution = asp[0]
                        best = self.f(neighbourhood(solution, asp[1], asp[2]))
                    # generujemy nowych sasiadow dopiki nie beda zabronieni
                    case "5":
                        while True:
                            while True:
                                s = random.randint(0, len(self.matrix) - 1)
                                e = random.randint(0, len(self.matrix) - 1)
                                if s != e:
                                    break
                            if s > e:
                                s, e = e, s

                            if not (s,e) in tabooList.queueToArray():
                                NwithoutTaboo.append((s,e))
                                break
                    case _:
                        return solution

            # wybor najlepszego rozwiazania
            if len(NwithoutTaboo)>0:
                pi = min(NwithoutTaboo, key=lambda t: self.f(neighbourhood(copy(solution), t[0], t[1])))
                tabooList.push((pi[0], pi[1]))
                neighbour = neighbourhood(solution, pi[0], pi[1])
                new = self.f(neighbour)
                # sprawdzenie czy najlepsze rozwiazanie jest gorsze niz to ktore bylo najlepsze przed usunieciem
                # jesli tak dodajemy jako kryterium aspiracji
                if new > self.f(neighbourhood(solution, bestTaboo[0], bestTaboo[1])):
                    aspirationList.push((solution, bestTaboo[0], bestTaboo[1]))
                if new < best:
                    solution = neighbour
                    best = new
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


# full = Full()
# full.random(5)
# # full.print_instance()
# # def tabuSearch(self, start, arg, neighbourhood, size=100, tabuSize=10, stop=10000, decision="1"):
# full.print_instance()
# startingArray = [i for i in range(5)]
# random.shuffle(startingArray)
# print(startingArray)
# print_result(full.tabuSearch(copy(startingArray), 5, invert, 5, 10, 10, "4"))
# print(startingArray)
# print_result(full.tabuSearch(copy(startingArray), 5, swap,   5, 10, 10, "4"))
# print(startingArray)
# print_result(full.tabuSearch(copy(startingArray), 5, insert, 5, 10, 10, "4"))



def compareIIS():
    # medota porównująca co jest lepsze: invert insert czy swap?
    # full = Full()
    full = Euc2D()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    invertResults = []
    insertResults = []
    swapResults = []
    
    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)
        start = time.time()
        full.tabuSearch(copy(startingArray), 5, invert, 5, 10, 10, "4")
        end = time.time()
        print(end - start)
        invertResults.append(end - start)

        start = time.time()
        full.tabuSearch(copy(startingArray), 5, insert, 5, 10, 10, "4")
        end = time.time()
        print(end - start)
        insertResults.append(end - start)

        start = time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "4")
        end = time.time()
        print(end - start)
        swapResults.append(end - start)

    print(invertResults)
    print(insertResults)
    print(swapResults)

    chart = Charts("Porównanie invert'a, insert'a oraz swap'a"," rozmiar instancji", "czas")
    chart.load(instanceSize, invertResults, "red", "invert")
    chart.load(instanceSize, insertResults, "green", "insert")
    chart.load(instanceSize, swapResults, "blue", "swap")
    chart.plot()

# compareIIS()

def compareIISres():
    # medota porównująca co jest lepsze: invert insert czy swap?
    # full = Full()
    full = Euc2D()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    invertResults = []
    insertResults = []
    swapResults = []
    
    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)
        s=full.tabuSearch(copy(startingArray), 5, invert, 5, 10, 10, "4")
        print(full.f(s))
        invertResults.append(full.f(s))

        s=full.tabuSearch(copy(startingArray), 5, insert, 5, 10, 10, "4")
        print(full.f(s))
        insertResults.append(full.f(s))

        s=full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "4")
        print(full.f(s))
        swapResults.append(full.f(s))

    print(invertResults)
    print(insertResults)
    print(swapResults)

    chart = Charts("Porównanie invert'a, insert'a oraz swap'a"," rozmiar instancji", "f")
    chart.load(instanceSize, invertResults, "red", "invert")
    chart.load(instanceSize, insertResults, "green", "insert")
    chart.load(instanceSize, swapResults, "blue", "swap")
    chart.plot()

# compareIISres()

def compareSizeOfTabuList():
    # metoda sprawdzająca jak zmienia się złożoność czasowa od rozmaiaru tablicy tab'u
    # full = Full()
    full = Euc2D()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    timeWithSmallSize = []
    timeWithMediumSize = []
    timeWithBigSize = []
    timeWith4Size = []
    timeWith6Size = []
    timeWith7Size = []

    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)

        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "4")
        # print("S:", s)
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWithSmallSize.append(full.f(s))
        # timeWithSmallSize.append((end-start))
        timeWithSmallSize.append(full.f(s) / (end-start))

        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 4, 10, 10, "4")
        # print("S:", s)
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWith4Size.append(full.f(s))
        # timeWith4Size.append((end-start))
        timeWith4Size.append(full.f(s) / (end-start))

        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 6, 10, 10, "4")
        # print("S:", s)
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWith6Size.append(full.f(s))
        # timeWith6Size.append((end-start))
        timeWith6Size.append(full.f(s) / (end-start))

        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 7, 10, 10, "4")
        # print("S:", s)
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWith7Size.append(full.f(s))
        # timeWith7Size.append((end-start))
        timeWith7Size.append(full.f(s) / (end-start))
    
        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 15, 10, 10, "4")
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWithMediumSize.append(full.f(s))
        # timeWithMediumSize.append((end-start))
        timeWithMediumSize.append(full.f(s) / (end-start))

        start = time.time()
        s = full.tabuSearch(copy(startingArray), 5, swap, 30, 10, 10, "4")
        print(full.f(s))
        end = time.time()
        print(end - start)
        # timeWithBigSize.append(full.f(s))
        # timeWithBigSize.append((end-start))
        timeWithBigSize.append(full.f(s) / (end-start))

    print(timeWithSmallSize)
    print(timeWithMediumSize)
    print(timeWithBigSize)

    chart = Charts("Porównanie na rozmiar tablicy taby"," rozmiar instancji", "czas/f(koszt)")
    chart.load(instanceSize, timeWith4Size, "yellow", "tabuSize: 4")
    chart.load(instanceSize, timeWithSmallSize, "red", "tabuSize: 5")
    chart.load(instanceSize, timeWith6Size, "magenta", "tabuSize: 6")
    chart.load(instanceSize, timeWith7Size, "black", "tabuSize: 7")
    chart.load(instanceSize, timeWithMediumSize, "green", "tabuSize: 15")
    chart.load(instanceSize, timeWithBigSize, "blue", "tabuSize: 30")
    chart.plot()

compareSizeOfTabuList()

def compareNoMovesOptionsresult():
    # metoda sprawdzająca jak zmienia się f celu w zależności od wyboru przy braku rozwiązań
    full = Full()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    opt1 = []
    opt2 = []
    opt3 = []
    opt4 = []
    opt5 = []

    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)

        s = full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "1")
        print(full.f(s))
        opt1.append(full.f(s))
    
        s = full.tabuSearch(copy(startingArray), 5, swap, 15, 10, 10, "2")
        print(full.f(s))
        opt2.append(full.f(s))


        s = full.tabuSearch(copy(startingArray), 5, swap, 30, 10, 10, "3")
        print(full.f(s))
        opt3.append(full.f(s))

        
        s = full.tabuSearch(copy(startingArray), 5, swap, 30, 10, 10, "4")
        print(full.f(s))
        opt4.append(full.f(s))

        
        s = full.tabuSearch(copy(startingArray), 5, swap, 30, 10, 10, "5")
        print(full.f(s))
        opt5.append(full.f(s))

    print(opt1)
    print(opt2)
    print(opt3)

    chart = Charts("Porównanie na wybór przy braku rozwiązania"," rozmiar instancji", "f(koszt)")
    chart.load(instanceSize, opt1, "red", "1")
    chart.load(instanceSize, opt2, "green", "2")
    chart.load(instanceSize,opt3, "blue", "3")
    chart.load(instanceSize,opt4, "yellow", "4")
    chart.load(instanceSize,opt5, "magenta", "5")
    chart.plot()

# compareNoMovesOptionsresult()

def compareNoMovesOptions():
    # metoda sprawdzająca jak zmienia się czas w zależności od wyboru przy braku rozwiązań
    full = Full()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    opt1 = []
    opt2 = []
    opt3 = []
    opt4 = []
    opt5 = []

    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)

        start=time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "1")
        end=time.time()
        print(end-start)
        opt1.append(end-start)
    
        start=time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "2")
        end=time.time()
        print(end-start)
        opt2.append(end-start)


        start=time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "3")
        end=time.time()
        print(end-start)
        opt3.append(end-start)
        
        start=time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "4")
        end=time.time()
        print(end-start)
        opt4.append(end-start)

        
        start=time.time()
        full.tabuSearch(copy(startingArray), 5, swap, 5, 10, 10, "5")
        end=time.time()
        print(end-start)
        opt5.append(end-start)

    print(opt1)
    print(opt2)
    print(opt3)

    chart = Charts("Porównanie na wybór przy braku rozwiązania"," rozmiar instancji", "czas")
    chart.load(instanceSize, opt1, "red", "1")
    chart.load(instanceSize, opt2, "green", "2")
    chart.load(instanceSize,opt3, "blue", "3")
    chart.load(instanceSize,opt4, "yellow", "4")
    chart.load(instanceSize,opt5, "magenta", "5")
    chart.plot()

# compareNoMovesOptions()

def compareSTOP():
    # medota porównująca co jest lepsze: invert insert czy swap?
    full = Full()
    instanceSize = [5, 10,15, 20,25, 30, 35]
    iterations = []
    iterationsNoChange = []
    time = []
    
    for currentInstatnceSize in instanceSize:
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)

        s=full.tabuSearch(copy(startingArray), 5, invert, 5, 10, 10, "1")
        print(full.f(s))
        iterations.append(full.f(s))

        s=full.tabuSearch1(copy(startingArray), 5, insert, 5, 10, 10, "1")
        print(full.f(s))
        iterationsNoChange.append(full.f(s))

        s=full.tabuSearch2(copy(startingArray), 5, swap, 5, 10, 10, "1")
        print(full.f(s))
        time.append(full.f(s))

    print(iterations)
    print(iterationsNoChange)
    print(time)

    chart = Charts("Porównanie warunku stopu"," rozmiar instancji", "f")
    chart.load(instanceSize, iterations, "red", "ilość iteracji")
    chart.load(instanceSize, iterationsNoChange, "green", "ilość iteracji bez zmiany")
    chart.load(instanceSize, time, "blue", "czas")
    chart.plot()

def compareSTOP2():
    # medota porównująca co jest lepsze: invert insert czy swap?
    full = Full()
    instanceSize = [5, 10, 15 ,20, 25 ] #, 30 ] #, 35]
    iterations = []
    iterationsNoChange = []
    timeArray = []
    
    for currentInstatnceSize in instanceSize:
        print(currentInstatnceSize)
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)
        
        start = time.time()
        s=full.tabuSearch(copy(startingArray), 5, invert, 5, 10, 10, "1")
        end = time.time()
        print(full.f(s))
        iterations.append(full.f(s) / (end - start))

        start = time.time()
        s=full.tabuSearch1(copy(startingArray), 5, insert, 5, 10, 10, "1")
        end = time.time()
        print(full.f(s))
        iterationsNoChange.append(full.f(s) / (end - start))
        
        start = time.time()
        s=full.tabuSearch2(copy(startingArray), 5, swap, 5, 10, 100, "1")
        end = time.time()
        print(full.f(s))
        timeArray.append(full.f(s) / (end - start))

    print(iterations)
    print(iterationsNoChange)
    print(timeArray)

    chart = Charts("Porównanie warunku stopu"," rozmiar instancji", "f")
    chart.load(instanceSize, iterations, "red", "ilość iteracji")
    chart.load(instanceSize, iterationsNoChange, "green", "ilość iteracji bez zmiany")
    chart.load(instanceSize, timeArray, "blue", "czas")
    chart.plot()

def timeComplexity():
    full = Full()
    instanceSize = [5, 10, 15 ,20, 25 ]
    timeComp = []
    # x2 = []

    for currentInstatnceSize in instanceSize:
        print(currentInstatnceSize)
        full.random(currentInstatnceSize)
        startingArray = [i for i in range(currentInstatnceSize)]
        random.shuffle(startingArray)

        start = time.time()
        # def tabuSearch(self, start, arg, neighbourhood, size=100, tabuSize=10, stop=10000, decision="1"):
        s=full.tabuSearch(copy(startingArray), 10, swap, 20, 40, 20, "2")
        end = time.time()
        # print(full.f(s))
        timeComp.append(end - start)


        # x2.append(currentInstatnceSize ** 2 / 80000)



    chart = Charts("time complexity"," rozmiar instancji", "time")
    chart.load(instanceSize, timeComp, "red", "czas")
    # chart.load(instanceSize, x2, "blue", "O(x^2)")
    chart.plot()

# timeComplexity()


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
