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
from pyparsing import Char, col
import tsplib95
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
from Charts import Charts
from heapq import *
import queue
from collections import deque
from operator import itemgetter

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

def PMX(parent1, parent2, size):
    cut = random.sample(range(0, size), 2)
    cut1 = cut[0]
    cut2 = cut[1]
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    fragment1 = parent1.genotyp[cut1:cut2]
    fragment2 = parent2.genotyp[cut1:cut2]
    map1 = dict()
    map2 = dict()
    for i in range(len(fragment1)):
        map1[fragment1[i]] = fragment2[i]
        map2[fragment2[i]] = fragment1[i]
    child1 = parent1.genotyp[:cut1] + fragment2 + parent1.genotyp[cut2:]
    child2 = parent2.genotyp[:cut1] + fragment1 + parent2.genotyp[cut2:]
    for i in range(cut1):
        while child1.count(child1[i])>1:
            child1[i] = map2.get(child1[i],child1[i])
        while child2.count(child2[i]) > 1:
            child2[i] = map1.get(child2[i], child2[i])
    for i in range(cut2, len(child1)):
        while child1.count(child1[i]) > 1:
            child1[i] = map2.get(child1[i], child1[i])
        while child2.count(child2[i]) > 1:
            child2[i] = map1.get(child2[i], child2[i])
    return child1,child2

def OX(parent1, parent2, size):
    cut = random.sample(range(1, size), 2)
    cut1 = cut[0]
    cut2 = cut[1]
    if cut1 > cut2:
        cut1, cut2 = cut2, cut1
    fragment = parent1.genotyp[cut1:cut2]
    parent2_copy = deepcopy(parent2)
    for i in fragment:
        parent2_copy.genotyp.remove(i)
    child1 = parent2_copy.genotyp[:cut1] + fragment + parent2_copy.genotyp[cut1:]
    fragment = parent2.genotyp[cut1:cut2]
    parent1_copy = deepcopy(parent1)
    for i in fragment:
        parent1_copy.genotyp.remove(i)
    child2 = parent1_copy.genotyp[:cut1] + fragment + parent1_copy.genotyp[cut1:]
    return child1, child2


def HX(parent1, parent2, size):
    onepointcut = random.randint(0, size)
    # child1 = parent1.genotyp[:onepointcut] + parent2.genotyp[onepointcut:]
    # child2 = parent2.genotyp[:onepointcut] + parent1.genotyp[onepointcut:]
    child1 = parent1.genotyp[:onepointcut] + missingNumber(parent1.genotyp[:onepointcut],
                                                           len(parent1.genotyp))
    child2 = parent2.genotyp[:onepointcut] + missingNumber(parent2.genotyp[:onepointcut],
                                                           len(parent2.genotyp))

    return child1, child2


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
        # sizeN to jak wielkie ma by?? s??siedztwo

        # rozwi??zanie pocz??tkowe
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

            # Czy wygenerowane s??sieddzctwo posiada lepsze PI
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
        while iterationNoChange < stop:
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

                            if not (s, e) in tabooList.queueToArray():
                                NwithoutTaboo.append((s, e))
                                break
                    case _:
                        return solution

            # wybor najlepszego rozwiazania
            if len(NwithoutTaboo) > 0:
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
                    print(iteration, new, end=" ")  # self.PRD(39, solution))
                    print(((new - f_best) / f_best) * 100)
                    solution = neighbour
                    best = new
                else:
                    iterationNoChange += 1

        return solution

    def generic(self, listOfDecision, start, arg, mutation, k, numberOfIndividuals, maxGeneration, preIndividualList=None):
        global optcycle
        # step 1: wygenerowanie populacji pocz??tkowej
        individuals_list = list()
        match listOfDecision[0]:
            case 1:
                for i in range(numberOfIndividuals):
                    x = [j for j in range(len(self.matrix))]
                    random.shuffle(x)
                    individuals_list.append(Individual(x, self.f(x)))
            case 2:
                if callable(start):
                    for i in range(numberOfIndividuals):
                        x = start(arg)
                        individuals_list.append(Individual(x, self.f(x)))
                else:
                    return "error"
            case 3: # predefiniowna lista 
                c = preIndividualList
                for i in c:
                    individuals_list.append(Individual(i, self.f(i)))
            case _:
                return "error"

        generation = 0
        generationWithoutChanges = 0 

        bestresult = min(individuals_list, key=lambda x: x.fenotyp).fenotyp
        bestresultlist = list()
        print(generation, bestresult, round(((bestresult - optcycle) / optcycle) * 100,2), "%")
        bestresultlist.append([generation, bestresult, round(((bestresult - optcycle) / optcycle) * 100,2)])

        startTime = time.time()

        # while pokolenie
        # while time.time()-startTime<stop:
        # while generation < maxGeneration:
        while generationWithoutChanges < maxGeneration:
            parents_list = list()
            for i in range(int(numberOfIndividuals / 2)):
                parent1 = None
                parent2 = None
                # step 2: selekcja
                match listOfDecision[1]:
                    case 1:  # losowa - ka??dy osobnik ma r??wne szanse bycia wybranym
                        while True:
                            parent1index = random.randint(0, len(individuals_list) - 1)
                            parent2index = random.randint(0, len(individuals_list) - 1)

                            if parent1index != parent2index:
                                break
                        parent1 = individuals_list[parent1index]
                        parent2 = individuals_list[parent2index]
                        parents_list.append((parent1, parent2))
                    case 2:  # ruletka
                        probability = list()
                        for i in individuals_list:
                            probability.append(i.fenotyp)
                        parent = random.choices(individuals_list, probability, k=2)
                        parents_list.append((parent[0], parent[1]))

                    case 3:  # turniej
                        rand_indiv = list()
                        for _ in range(k):
                            rand_indiv.append(individuals_list[random.randint(0, len(individuals_list) - 1)])
                        parent1 = min(rand_indiv, key=lambda x: x.fenotyp)
                        rand_indiv = []
                        for _ in range(k):
                            while True:
                                x=individuals_list[random.randint(0, len(individuals_list) - 1)]
                                if x.genotyp!=parent1.genotyp:
                                    break
                            rand_indiv.append(x)
                        parent2 = min(rand_indiv, key=lambda x: x.fenotyp)
                        parents_list.append((parent1, parent2))
                    case _:
                        return "error"

            individuals_list = []
            for parents in parents_list:
                parent1 = parents[0]
                parent2 = parents[1]

                #print(parent1.genotyp, parent2.genotyp)
                # step 3: krzy??owanie
                match listOfDecision[2]:
                    case 1:  # Half Crossover, HX
                        child1, child2 = HX(parent1, parent2, len(self.matrix))
                        individuals_list.append(Individual(child1, self.f(child1)))
                        individuals_list.append(Individual(child2, self.f(child2)))
                    case 2:  # Order Crossover, OX
                        child1, child2 = OX(parent1, parent2, len(self.matrix))
                        individuals_list.append(Individual(child1, self.f(child1)))
                        individuals_list.append(Individual(child2, self.f(child2)))
                    case 3:  # Partially Mapped Crossover PMX
                        child1, child2 = PMX(parent1, parent2, len(self.matrix))
                        individuals_list.append(Individual(child1, self.f(child1)))
                        individuals_list.append(Individual(child2, self.f(child2)))
                    case _:
                        return "error"

            # step 4
            for individual in individuals_list:
                if random.random() < listOfDecision[3]:
                    if callable(mutation):
                        mutation(individual.genotyp, random.randint(0, len(self.matrix)-1),
                                 random.randint(0, len(self.matrix)-1))

            generation += 1
            generationWithoutChanges += 1

            currentresult = min(individuals_list, key=lambda x: x.fenotyp).fenotyp
            if currentresult < bestresult:
                print(generation, currentresult, round(((currentresult - optcycle) / optcycle) * 100,2), "%")
                bestresultlist.append([generation, currentresult, round(((currentresult - optcycle) / optcycle) * 100,2)])
                bestresult = currentresult
                generationWithoutChanges = 0

                # if currentresult == optcycle:
                #     break


        # for i in individuals_list:
        #     print(i.genotyp,i.fenotyp)
        result = min(individuals_list, key=lambda x: x.fenotyp)
        return result.genotyp, bestresultlist


def missingNumber(lis, max):
    x = [i for i in range(max)]
    for l in lis:
        if l in x:
            x.remove(l)
    random.shuffle(x)
    return x


class Individual:
    genotyp = list()  # ci??g kodowy gen??w reprezentuj??cych rozwi??zanie
    fenotyp = 0

    def __init__(self, gen, fen):  # Individual([1,2,3....,n], Graph.f([1,2,3,...,n]))
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
                self.matrix[i][j] = int(sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))  # cz?????? rzeczywista liczby zespolonej

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

def generateData(numberOfIndividuals, size):
    c = list()
    for i in range(numberOfIndividuals):
        x = [j for j in range(size)]
        random.shuffle(x)
        c.append(x)
    return c


full=Full()
# full.random(10)
full.load("gr120.tsp")
optcycle = 6942 # 2085
data = generateData(100, len(full.matrix))

listOfDecision = [3, 3, 1, 0.1]
# param1: 1 - random start cycle 2 - full.twoOPT 3 - predef individual list
# param2: 1 - losowa 2 - ruletka 3 - turniej
# param3: 1 - HX, 2 - OX, 3 - PMX
# param4: propability
_, dataHX = full.generic(listOfDecision, full.twoOPT, 15, swap, 10, 100, 1000, data)
listOfDecision = [3, 3, 2, 0.1]
print("\n")
_, dataOX = full.generic(listOfDecision, full.twoOPT, 15, swap, 10, 100, 1000, data)
listOfDecision = [3, 3, 3, 0.1]
print("\n")
_, dataPMX = full.generic(listOfDecision, full.twoOPT, 15, swap, 10, 100, 1000, data)

def splitter(data):
    x = list(map(itemgetter(0), data))
    y = list(map(itemgetter(1), data))
    prd = list(map(itemgetter(2), data))
    return x, y, prd

xHX, yHX, prdHX = splitter(dataHX)
xOX, yOX, prdOX = splitter(dataOX)
xPMX, yPMX, prdPMX = splitter(dataPMX)

c = Charts("gr120", "generacja", "PRD")
c.load(xHX, prdHX, "red", "HX")
c.load(xOX, prdOX, "green", "OX")
c.load(xPMX, prdPMX, "blue", "PMX")
c.plot(annotate=True)