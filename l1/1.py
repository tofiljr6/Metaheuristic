import random
from math import sqrt

import tsplib95
from matplotlib import pyplot as plt


def full_matrix(filename):
    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())


    matrix = [[None for i in range(len(nodeList))] for j in range(len(nodeList))]

    for i in nodeList:
        for j in nodeList:
            matrix[i][j]=problem.get_weight(i,j)

    return matrix

def lower_diag_row(filename):
    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())
    matrix = [[0 for i in range(j+1)] for j in range(len(nodeList))]

    for i in nodeList:
        for j in range(1,i):
            matrix[i-1][j-1] = problem.get_weight(i, j)

    return matrix

# wczytywanie euc
def euc_2d(filename):

    problem = tsplib95.load(filename)
    nodeList = list(problem.get_nodes())
    edgesList = list(problem.get_edges())

    global x,y
    x=[]
    y=[]

    matrix = [[None for i in range(j+1)] for j in range(len(nodeList))]

    for i in range(1,len(nodeList)+1):
        x.append(problem.node_coords[i][0])
        y.append(problem.node_coords[i][1])

    for i in range(len(edgesList)):
        edge = edgesList[i]
        try:
            matrix[edge[0] - 1][edge[1] - 1] = problem.get_weight(*edge)
        except IndexError:
            pass

    return matrix

# losowa macierz pelna
def random_full(size):
    matrix = [[None for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j]=random.randint(0,1000)
    return matrix

# losowa macierz lustrzana
def random_lower(size):
    matrix = [[0 for i in range(j + 1)] for j in range(size)]
    for i in range(size):
        for j in range(i):
            matrix[i][j]=random.randint(0,1000)
    return matrix

# losowa macierz eukidesowa 2D
def random_euc_2d(size):
    matrix = [[0 for i in range(j+1)] for j in range(size)]
    global x,y
    x=[]
    y=[]
    for i in range(size):
        x.append(random.randint(0, 1000))
        y.append( random.randint(0, 1000))

    for i in range(size):
        for j in range(i):
            x1, y1 =(x[i],y[i])
            x2, y2 =(x[j],y[j])
            matrix[i][j] = int(sqrt((x2-x1) ** 2 + (y2-y1) ** 2)) # część rzeczywista liczby zespolonej

    return matrix

# na razie zwraca byle jakie rozwiazanie
def calculate_result(matrix):
    result=[]
    for i in range(len(matrix)):
        result.append(i)
    return result

# wypisanie instancji
def print_instance(instance):
    for i in range(len(instance)):
        print(instance[i])


# wypisanie trasy
def print_result(result):
    for i in range(len(result)):
        print(result[i]+1,"-> ",end="")
    print(result[0]+1)

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
    if len(matrix[0])==1:
        for c in range(len(cycle) - 1):
            sum += matrix[cycle[c + 1]][cycle[c]]
        sum += matrix[cycle[0]][cycle[len(cycle) - 1]]
    else:
        for c in range(len(cycle)-1):
            sum += matrix[cycle[c]][cycle[c+1]]
        sum += matrix[cycle[len(cycle)-1]][cycle[0]]
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

def nearest_neighbour(matrix,position):
    result=[position]
    i=0
    while i<len(matrix)-1:
        min=99999
        index=0
        for j in range(len(matrix[position])):
            if j not in result:
                if matrix[position][j]<min:
                    min=matrix[position][j]
                    index=j
        print(matrix[position][index])
        position=index
        result.append(position)
        i+=1
    return result

def invert(vector,i,j):
    vector[i:j+1]=vector[i:j+1][::-1]
    return vector


