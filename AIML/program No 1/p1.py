from heuristicsearch.a_star_search import AStar
adjacency_list = {
    'S': [('A', 1), ('G', 10)], 
    'A': [('B', 2), ('C', 1)],
    'B': [('D', 5)],
    'C': [('D', 3), ('G', 4)],
    'D': [('G', 2)]
}
heuristics = {'S': 1, 'A': 1, 'B': 1, 'C': 1, 'D': 1, 'G': 1}
graph = AStar(adjacency_list, heuristics)
graph.apply_a_star(start= 'S', stop= 'G')
