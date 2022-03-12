import random
import numpy
import tsplib95
from matplotlib import pyplot as plt


def full_matrix(filename):
    """ The methods reads graphs information such as start and end vertex, and weight of this edge.
        Param: filename - name of file in .tsp or .atsp extension.
        Return: returns a full neighbourhood matrix.
                IMPORTANT: vertices in graph counts from 1.
                If we want to get for example vertex, which has cords (1, 1), we should write matrix[0][0]
    """
    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())
    edgesList = list(problem.get_edges())

    matrix = [[None for i in range(0, len(nodeList))] for j in range(0, len(nodeList))]

    for i in range(len(edgesList)):
        edge = edgesList[i]
        matrix[edge[0]-1][edge[1]-1] = problem.get_weight(*edge)

    return matrix

def lower_diag_row(filename):
    """ The methods reads graphs information such as start and end vertex, and weight of this edge.
        Param: filename - name of file in .tsp or .atsp extension.
        Return: returns a lower  diagonal row of neighbourhood matrix.
                IMPORTANT: vertices in graph counts from 1.
                If we want to get for example vertex, which has cords (1, 1), we should write matrix[0][0]
    """
    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())
    edgesList = list(problem.get_edges())

    matrix = [[None for i in range(0, j+1)] for j in range(0, len(nodeList))]

    for i in range(len(edgesList)):
        edge = edgesList[i]
        try:
            matrix[edge[0]-1][edge[1]-1] = problem.get_weight(*edge)
        except IndexError:
            pass

    return matrix

# wczytywanie euc
def euc_2d(filename):
    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())
    edgesList = list(problem.get_edges())

    global x,y
    x=[]
    y=[]

    matrix = [[None for i in range(0, len(nodeList))] for j in range(0, len(nodeList))]

    for i in range(1,len(nodeList)+1):
        x.append(problem.node_coords[i][0])
        y.append(problem.node_coords[i][1])

    for i in range(len(edgesList)):
        edge = edgesList[i]
        matrix[edge[0]-1][edge[1]-1] = problem.get_weight(*edge)

    return matrix

# losowa macierz pelna
def random_full(size):
    matrix = numpy.zeros((size,size))
    for i in range(size):
        for j in range(size):
            matrix[i][j]=random.randint(0,1000)
    return matrix

# losowa macierz lustrzana
def random_lower(size):
    matrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(i):
            value=random.randint(0,1000)
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix

# na razie zwraca byle jakie rozwiazanie
def calculate_result(matrix):
    result=[]
    for i in range(len(matrix)):
        result.append(i)
    return result

# wypisanie trasy
def print_result(result):
    for i in range(len(result)):
        print(i,"-> ",end="")
    print(result[0])

# trasa graficznie
def result_euc(result):
    plt.scatter(x,y)
    for i in range(len(x)):
        plt.text(x[i],y[i],i)
    plt.plot([x[i] for i in range(len(result))], [y[i] for i in range(len(result))], color="red")
    plt.show()

#TODO
def f(result):
    return len(result)

# blad wzgledny
def PRD(best_result,result):
    f_res=f(result)
    f_best=f(best_result)
    return ((f_res-f_best)/f_best)*100





