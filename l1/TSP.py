from operator import le
import re
import sys
import os
from turtle import color
from numpy import mat
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
            print(f_res, f_best)
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
        while iterationNoChange<stop:
        # while iteration < stop:
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
                    # f_res = self.f(neighbour)
                    f_best = 6942
                    print(iteration, new, end=" ") # self.PRD(39, solution))
                    print(((new - f_best) / f_best) * 100)
                    solution = neighbour
                    best = new
                else:
                    iterationNoChange += 1
                
        return solution

    def generic(self, listOfDecision, numberOfIndividuals):
        individuals_list = list()

        # step 1: wygenerowanie populacji początkowej
        match listOfDecision[0]: 
            case 1: # random
                for i in range(numberOfIndividuals):
                    x = [j for j in range(numberOfIndividuals)]
                    random.shuffle(x)
                    individuals_list.append(Individual(x, self.f(x)))
            case 2: # stosując metaheurystyke
                pass
            case _:
                return "error"


        # while pokolenie
        
        parent1 = None
        parent2 = None
        # step 2: selekcja 
        match listOfDecision[1]:
            case 1: # losowa - każdy osobnik ma równe szanse bycia wybranym
                print("case 1")
                while True:
                    parent1index = random.randint(0, len(individuals_list)-1)
                    parent2index = random.randint(0, len(individuals_list)-1)

                    if parent1index != parent2index:
                        break
                parent1 = individuals_list[parent1index]
                parent2 = individuals_list[parent2index]
            case 2: # ruletka
                pass
            case 3: # turniej
                pass
            case _:
                return "error"


        # step 3: krzyżowanie
        match listOfDecision[2]:
            case 1: # Half Crossover, HX
                onepointcut = random.randint(0, len(self.matrix))
                print(onepointcut)
                child1 = parent1.genotyp[:onepointcut] + parent2.genotyp[onepointcut:]
                child2 = parent2.genotyp[:onepointcut] + parent1.genotyp[onepointcut:]
                print(parent1.genotyp)
                print(parent2.genotyp)
                print(child1)
                print(child2)

                individuals_list.remove(parent1)
                individuals_list.remove(parent2)
                individuals_list.append(Individual(child1, self.f(child1)))
                individuals_list.append(Individual(child2, self.f(child2)))

            case 1: # Order Crossover, OX
                pass
            case 1: # Partially Mapped Crossover PMX
                pass
            case _:
                return "error"




class Individual:
    genotyp = list() # ciąg kodowy genów reprezentujących rozwiązanie
    fenotyp = 0

    def __init__(self, gen, fen): # Individual([1,2,3....,n], Graph.f([1,2,3,...,n]))
        self.genotyp = gen
        self.fenotyp = fen


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

full = Full()
full.random(5)
listOfDecision = [1, 1, 1] # patrz: krok 1, miał do wyboru (1) losowa, (2) stosując heurystyke 

full.generic(listOfDecision, 5)
print("sss")