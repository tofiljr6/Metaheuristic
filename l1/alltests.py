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

# compareSizeOfTabuList()

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



# instace = Lower()
# instace.load("gr120.tsp")
# s = instace.tabuSearch(copy(instace.krandom),1, swap, 750, 15, 3000, "5")
# # s = instace.tabuSearch(copy(instace.krandom),10, swap, 21, 7, 10000, "2")
# print_result(s)

# d = copy(instace.nearest_neighbour(100))
# print(instace.PRD(6942, d))











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
