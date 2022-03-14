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

# losowa macierz eukidesowa 2D
def random_euc_2d(size):
    matrix = numpy.zeros((size, size))
    matrixCoordinates = numpy.empty((size), dtype=object)
    for i in range(size):
        x, y = random.randint(0, 1000), random.randint(0, 1000)
        matrixCoordinates[i] = (x, y)

    for i in range(size):
        for j in range(i):
            coods1 = matrixCoordinates[i]
            coods2 = matrixCoordinates[j]
            x1, y1 = coods1
            x2, y2 = coods2
            matrix[i][j] = sqrt((x2-x1) ** 2 + (y2-y1) ** 2).real # część rzeczywista liczby zespolonej

    return matrix

# na razie zwraca byle jakie rozwiazanie
def calculate_result(matrix):
    result=[]
    for i in range(len(matrix)):
        result.append(i)
    return result

# wypisanie instancji
def print_instance(instance):
    print(instance)

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

# funckja celu
def f(cycle, matrix):
    sum = 0
    for c in range(len(cycle)-1):
        sum += matrix[cycle[c]-1][cycle[c+1]-1]
    sum += matrix[cycle[len(cycle)-1]-1][cycle[0]-1]

    return sum

# blad wzgledny
def PRD(best_result,result, matrix):
    if type(best_result) == int:
        f_res=f(result, matrix)
        f_best=best_result
        return ((f_res-f_best)/f_best)*100
    elif type(best_result) == list:
        f_res=f(result, matrix)
        f_best=f(best_result, matrix)
        return ((f_res-f_best)/f_best)*100

result_euc(calculate_result(euc_2d('./berlin52.tsp')))