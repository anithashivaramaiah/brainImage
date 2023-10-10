def numOfTriangles(input_graph):
    vertex = len(input_graph)
    num = []
    temp = list()
    for i in range(vertex):
        for j in range(vertex):
            for k in range(vertex):
                if (i != j and i != k and j != k and
                        input_graph[i][j] and
                        input_graph[j][k] and
                        input_graph[k][i]):
                    new_graph = list(sorted([i, j, k]))
                    if new_graph not in temp:
                        temp.append(new_graph)
                        num.append([(i, j), (j, k), (k, i)])
    for i in num:
        print(i)


graph1 = [[0, 1, 0, 0],
          [0, 0, 1, 0],
          [1, 0, 0, 1],
          [0, 1, 0, 0]]

graph2 = [[0, 1, 0],
          [1, 0, 1],
          [1, 0, 0]]

print("The number of triangles for first graph are: ")
numOfTriangles(graph1)
print("The number of triangles for second graph are: ")
numOfTriangles(graph2)

# OUTPUT:
# The number of triangles for first graph are:
# [(0, 1), (1, 2), (2, 0)]
# [(1, 2), (2, 3), (3, 1)]
# The number of triangles for second graph are:
# [(0, 1), (1, 2), (2, 0)]
