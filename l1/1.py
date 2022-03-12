import tsplib95

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
