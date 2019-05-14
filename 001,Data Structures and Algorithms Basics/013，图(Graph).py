# 第一部分、创建图:
# 1，矩阵表示法:
class Vertex:
    def __init__(self, node):
        self.id = node
        # Mark all nodes unvisited        
        self.visited = False  

    def addNeighbor(self, neighbor, G):
        G.addEdge(self.id, neighbor)

    def getConnections(self, G):
        return G.adjMatrix[self.id]

    def getVertexID(self):
        return self.id

    def setVertexID(self, id):
        self.id = id

    def setVisited(self):
        self.visited = True

    def __str__(self):
        return str(self.id)

class Graph:
    def __init__(self, numVertices=10, directed=False):
        self.adjMatrix = [[None] * numVertices for _ in range(numVertices)]
        self.numVertices = numVertices
        self.vertices = []
        self.directed = directed
        for i in range(0, numVertices):
            newVertex = Vertex(i)
            self.vertices.append(newVertex)

    def addVertex(self, vtx, id):
        if 0 <= vtx < self.numVertices:
            self.vertices[vtx].setVertexID(id)

    def getVertex(self, n):
        for vertxin in range(0, self.numVertices):
            if n == self.vertices[vertxin].getVertexID():
                return vertxin
        return None

    def addEdge(self, frm, to, cost=0): 
        #print("from",frm, self.getVertex(frm))
        #print("to",to, self.getVertex(to))
        if self.getVertex(frm) is not None and self.getVertex(to) is not None:
            self.adjMatrix[self.getVertex(frm)][self.getVertex(to)] = cost
            if not self.directed:
                # For directed graph do not add this
                self.adjMatrix[self.getVertex(to)][self.getVertex(frm)] = cost  

    def getVertices(self):
        vertices = []
        for vertxin in range(0, self.numVertices):
            vertices.append(self.vertices[vertxin].getVertexID())
        return vertices

    def printMatrix(self):
        for u in range(0, self.numVertices):
            row = []
            for v in range(0, self.numVertices):
                row.append(str(self.adjMatrix[u][v]) if self.adjMatrix[u][v] is not None else '/')
            print(row)

    def getEdges(self):
        edges = []
        for v in range(0, self.numVertices):
            for u in range(0, self.numVertices):
                if self.adjMatrix[u][v] is not None:
                    vid = self.vertices[v].getVertexID()
                    wid = self.vertices[u].getVertexID()
                    edges.append((vid, wid, self.adjMatrix[u][v]))
        return edges
    
    def getNeighbors(self, n):
        neighbors = []
        for vertxin in range(0, self.numVertices):
            if n == self.vertices[vertxin].getVertexID():
                for neighbor in range(0, self.numVertices):
                    if (self.adjMatrix[vertxin][neighbor] is not None):
                        neighbors.append(self.vertices[neighbor].getVertexID())
        return neighbors
    
    def isConnected(self, u, v):
        uidx = self.getVertex(u) 
        vidx = self.getVertex(v)
        return self.adjMatrix[uidx][vidx] is not None
    
    def get2Hops(self, u):
        neighbors = self.getNeighbors(u)
        print(neighbors)
        hopset = set()
        for v in neighbors:
            hops = self.getNeighbors(v)
            hopset |= set(hops)
        return list(hopset)

if __name__ == '__main__':
	graph = Graph(6,True)
	graph.addVertex(0, 'a')
	graph.addVertex(1, 'b')
	graph.addVertex(2, 'c')
	graph.addVertex(3, 'd')
	graph.addVertex(4, 'e')
	graph.addVertex(5, 'f')
	graph.addVertex(6, 'g') # doing nothing here 
	graph.addVertex(7, 'h') # doing nothing here

	print(graph.getVertices())
	graph.addEdge('a', 'b', 1)  
	graph.addEdge('a', 'c', 2)
	graph.addEdge('b', 'd', 3)
	graph.addEdge('b', 'e', 4)
	graph.addEdge('c', 'd', 5)
	graph.addEdge('c', 'e', 6)
	graph.addEdge('d', 'e', 7)
	graph.addEdge('e', 'a', 8)
	print(graph.printMatrix())
	print(graph.getEdges())

# 2,列表表示法:
import sys
class Vertex:
    def __init__(self, node):
        self.id = node
        self.adjacent = {}
        # Set distance to infinity for all nodes
        self.distance = sys.maxsize
        # Mark all nodes unvisited        
        self.visited = False  
        # Predecessor
        self.previous = None

    def addNeighbor(self, neighbor, weight=0):
        self.adjacent[neighbor] = weight

    # returns a list 
    def getConnections(self): # neighbor keys
        return self.adjacent.keys()  

    def getVertexID(self):
        return self.id

    def getWeight(self, neighbor):
        return self.adjacent[neighbor]

    def setDistance(self, dist):
        self.distance = dist

    def getDistance(self):
        return self.distance

    def setPrevious(self, prev):
        self.previous = prev

    def setVisited(self):
        self.visited = True

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.adjacent])
    
    def __lt__(self, other):
        return self.distance < other.distance and self.id < other.id    

class Graph:
    def __init__(self, directed=False):
        # key is string, vertex id
        # value is Vertex
        self.vertDictionary = {}
        self.numVertices = 0
        self.directed = directed
        
    def __iter__(self):
        return iter(self.vertDictionary.values())

    def isDirected(self):
        return self.directed
    
    def vectexCount(self):
        return self.numVertices

    def addVertex(self, node):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(node)
        self.vertDictionary[node] = newVertex
        return newVertex

    def getVertex(self, n):
        if n in self.vertDictionary:
            return self.vertDictionary[n]
        else:
            return None

    def addEdge(self, frm, to, cost=0):
        if frm not in self.vertDictionary:
            self.addVertex(frm)
        if to not in self.vertDictionary:
            self.addVertex(to)

        self.vertDictionary[frm].addNeighbor(self.vertDictionary[to], cost)
        if not self.directed:
            # For directed graph do not add this
            self.vertDictionary[to].addNeighbor(self.vertDictionary[frm], cost)

    def getVertices(self):
        return self.vertDictionary.keys()

    def setPrevious(self, current):
        self.previous = current

    def getPrevious(self, current):
        return self.previous

    def getEdges(self):
        edges = []
        for key, currentVert in self.vertDictionary.items():
            for nbr in currentVert.getConnections():
                currentVertID = currentVert.getVertexID()
                nbrID = nbr.getVertexID()
                edges.append((currentVertID, nbrID, currentVert.getWeight(nbr))) # tuple
        return edges
    
    def getNeighbors(self, v):
        vertex = self.vertDictionary[v]
        return vertex.getConnections()

def graphFromEdgelist(E, directed=False):
    """Make a graph instance based on a sequence of edge tuples.
    Edges can be either of from (origin,destination) or
    (origin,destination,element). Vertex set is presume to be those
    incident to at least one edge.
    vertex labels are assumed to be hashable.
    """
    g = Graph(directed)
    V = set()
    for e in E:
        V.add(e[0])
        V.add(e[1])
        
    print("Vertex: ", V)

    verts = {}  # map from vertex label to Vertex instance
    for v in V:
        verts[v] = g.addVertex(v)
    print(g.vectexCount())

    for e in E:
        src = e[0]
        dest = e[1]
        cost = e[2] if len(e) > 2 else None
        g.addEdge(src, dest, cost)
    return g

if __name__ == '__main__':
	G = Graph(True)
	G.addVertex('a')
	G.addVertex('b')
	G.addVertex('c')
	G.addVertex('d')
	G.addVertex('e')
	G.addVertex('f')
	G.addEdge('a', 'b', 1)  
	G.addEdge('a', 'c', 1)
	G.addEdge('b', 'd', 1)
	G.addEdge('b', 'e', 1)
	G.addEdge('c', 'd', 1)
	G.addEdge('c', 'e', 1)
	G.addEdge('d', 'e', 1)
	G.addEdge('e', 'a', 1)
	print (G.getEdges())
	for k in G.getEdges():
	    print(k)

	E = (
	('SFO', 'LAX', 337), ('SFO', 'BOS', 2704), ('SFO', 'ORD', 1846),
	('SFO', 'DFW', 1464), ('LAX', 'DFW', 1235), ('LAX', 'MIA', 2342),
	('DFW', 'ORD', 802), ('DFW', 'MIA', 1121), ('ORD', 'BOS', 867),
	('ORD', 'JFK', 740), ('MIA', 'JFK', 1090), ('MIA', 'BOS', 1258), 
	('JFK', 'BOS', 187),
	)
	graph = graphFromEdgelist(E, True)
	for e in graph.getEdges():
	    print(e)

	for m in graph.getVertices():
	    print(m)

# 第二部分、搜索(DFS/BFS):
# 1，深度优先搜索(DFS)
# 递归法：
def dfs(G, currentVert, visited):
    visited[currentVert] = True  # mark the visited node 
    print("traversal: " + currentVert.getVertexID())
    for nbr in currentVert.getConnections():  # take a neighbouring node 
        if nbr not in visited:  # condition to check whether the neighbour node is already visited
            dfs(G, nbr, visited)  # recursively traverse the neighbouring node
    return 
 
def DFSTraversal(G):
    visited = {}  # Dictionary to mark the visited nodes 
    for currentVert in G:  # G contains vertex objects
        if currentVert not in visited:  # Start traversing from the root node only if its not visited 
            dfs(G, currentVert, visited)  # For a connected graph this is called only onc

# 迭代法：
def dfsIterative(G, start, dest):
    stack = [] # vertex
    visited = set() # vertex id
    parent = {} # vertex id
    stack.append(start)
    while len(stack) != 0:
        curr = stack.pop() # vertex
        print("visiting ", curr.getVertexID())
        if (curr.getVertexID() == dest.getVertexID()):
            return parent
        neighbors = G.getNeighbors(curr.getVertexID())
        for n in neighbors:
            id = n.getVertexID()
            visited.add(id)
            parent[id] = curr.getVertexID()
            stack.append(n)
    return None

if __name__ == '__main__':
	G = Graph(True)
	G.addVertex('a')
	G.addVertex('b')
	G.addVertex('c')
	G.addVertex('d')
	G.addVertex('e')
	G.addVertex('f')
	G.addEdge('a', 'b', 1)  
	G.addEdge('a', 'c', 1)
	G.addEdge('b', 'd', 1)
	G.addEdge('b', 'e', 1)
	G.addEdge('c', 'd', 1)
	G.addEdge('c', 'e', 1)
	G.addEdge('d', 'e', 1)
	G.addEdge('e', 'a', 1)
	G.addEdge('a', 'f', 1)
	print (G.getEdges())
	for k in G.getEdges():
	    print(k)

	start = G.getVertex('a')
	dest = G.getVertex('e')
	parent = dfsIterative(G, start, dest)
	print(parent)

# 2，广度优先搜索(BFS)：
from collections import deque

def bfs(G, start, dest):
    queue = deque() # vertex
    visited = set() # vertex id
    parent = {} # vertex id
    queue.append(start)
    while len(queue) != 0:
        curr = queue.popleft() # vertex
        print("visiting ", curr.getVertexID())
        if (curr.getVertexID() == dest.getVertexID()):
            return parent
        neighbors = G.getNeighbors(curr.getVertexID())
        for n in neighbors:
            id = n.getVertexID()
            visited.add(id)
            parent[id] = curr.getVertexID()
            queue.append(n)
    return None

if __name__ == '__main__':
	G = Graph(True)
	G.addVertex('a')
	G.addVertex('b')
	G.addVertex('c')
	G.addVertex('d')
	G.addVertex('e')
	G.addVertex('f')
	G.addEdge('a', 'b', 1)  
	G.addEdge('a', 'c', 1)
	G.addEdge('a', 'd', 1)
	G.addEdge('d', 'e', 1)
	G.addEdge('e', 'f', 1)
	print (G.getEdges())
	for k in G.getEdges():
	    print(k)
	start = G.getVertex('a')
	dest = G.getVertex('e')
	parent = bfs(G, start, dest)
	print(parent)

# 3，Dijkstra算法：
import heapq

def shortest(v, path):
    ''' make shortest path from v.previous'''
    if v.previous:
        path.append(v.previous.getVertexID())
        shortest(v.previous, path)
    return

def dijkstra(G, source, destination):
    print('''Dijkstra's shortest path''')
    # Set the distance for the source node to zero 
    source.setDistance(0)

    # Put tuple pair into the priority queue
    unvisitedQueue = [(v.getDistance(), v) for v in G]
    heapq.heapify(unvisitedQueue)

    while len(unvisitedQueue):
        # Pops a vertex with the smallest distance 
        uv = heapq.heappop(unvisitedQueue)
        current = uv[1]
        current.setVisited()

        # for next in v.adjacent:
        for next in current.adjacent:
            # if visited, skip
            if next.visited:
                continue
            newDist = current.getDistance() + current.getWeight(next)
            
            if newDist < next.getDistance():
                next.setDistance(newDist)
                next.setPrevious(current)
                print('Updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))
            else:
                print('Not updated : current = %s next = %s newDist = %s' \
                        % (current.getVertexID(), next.getVertexID(), next.getDistance()))

        # Rebuild heap
        # 1. Pop every item
        while len(unvisitedQueue):
            heapq.heappop(unvisitedQueue)
        # 2. Put all vertices not visited into the queue
        unvisitedQueue = [(v.getDistance(), v) for v in G if not v.visited]
        heapq.heapify(unvisitedQueue)

if __name__ == '__main__':
	G = Graph(True)
	G.addVertex('a')
	G.addVertex('b')
	G.addVertex('c')
	G.addVertex('d')
	G.addVertex('e')
	G.addEdge('a', 'b', 4)  
	G.addEdge('a', 'c', 1)
	G.addEdge('c', 'b', 2)
	G.addEdge('b', 'e', 4)
	G.addEdge('c', 'd', 4)
	G.addEdge('d', 'e', 4)

	for v in G:
	    for w in v.getConnections():
	        vid = v.getVertexID()
	        wid = w.getVertexID()
	        print('( %s , %s, %3d)' % (vid, wid, v.getWeight(w)))

	source = G.getVertex('a')
	destination = G.getVertex('e')    
	dijkstra(G, source, destination) 

	for v in G.vertDictionary.values():
	    print(source.getVertexID(), " to ", v.getVertexID(), "-->", v.getDistance())

	path = [destination.getVertexID()]
	shortest(destination, path)
	print ('The shortest path from a to e is: %s' % (path[::-1]))

# 第三部分、相关练习题：
# 1,迷宫(1):
def dfs1(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    return dfsHelper(matrix, start, dest, visited)
    
def dfsHelper(matrix, start, dest, visited):
    if matrix[start[0]][start[1]] == 1:
        return False
    
    if visited[start[0]][start[1]]:
        return False
    if start[0] == dest[0] and start[1] == dest[1]:
        return True
    
    visited[start[0]][start[1]] = True
    
    if (start[1] < len(matrix[0]) - 1):
        r = (start[0], start[1] + 1)
        if (dfsHelper(matrix, r, dest, visited)):
            return True
        
    if (start[1] > 0):
        l = (start[0], start[1] - 1)
        if (dfsHelper(matrix, l, dest, visited)):
            return True
        
    if (start[0] > 0):
        u = (start[0] - 1, start[1])
        if (dfsHelper(matrix, u, dest, visited)):
            return True
        
    if (start[0] < len(matrix[0]) - 1):
        d = (start[0] + 1, start[1])
        if (dfsHelper(matrix, d, dest, visited)):
            return True
            
    return False

def dfsIterative2(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    stack = []
    stack.append(start)
    visited[start[0]][start[1]] = True
    
    idxs = [[0,1], [0,-1], [-1,0], [1,0]]
    
    while len(stack) != 0:
        curr = stack.pop() # vertex
        if (curr[0] == dest[0] and curr[1] == dest[1]):
            return True

        for idx in idxs:
            x = curr[0] + idx[0]
            y = curr[1] + idx[1]
            
            if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0])):
                continue
            
            if (matrix[x][y] == 1):
                continue
                
            if (visited[x][y] == True):
                continue
            visited[x][y] = True
            stack.append((x, y))
            
    return False

from collections import deque

def bfs3(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    queue = deque()
    queue.append(start)
    visited[start[0]][start[1]] = True
    
    idxs = [[0,1], [0,-1], [-1,0], [1,0]]
    
    while len(queue) != 0:
        curr = queue.popleft() # vertex
        if (curr[0] == dest[0] and curr[1] == dest[1]):
            return True

        for idx in idxs:
            x = curr[0] + idx[0]
            y = curr[1] + idx[1]
            
            if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0])):
                continue
            
            if (matrix[x][y] == 1):
                continue
                
            if (visited[x][y] == True):
                continue
            visited[x][y] = True
            queue.append((x, y))
            
    return False

if __name__ == '__main__':
	matrix = [
	    [0, 0, 1, 0, 0],
	    [0, 0, 0, 0, 0],
	    [0, 0, 0, 1, 0],
	    [1, 1, 1, 1, 1],
	    [0, 0, 0, 0, 0]
	]

	start = (0, 0)
	dest  = (4, 4)
	print(bfs3(matrix, start, dest))

# 2,迷宫(2):
def dfs2(matrix, start, dest):
    visited = [[False] * len(matrix[0]) for i in range(len(matrix))]
    return dfsHelper2(matrix, start, dest, visited)
    
def dfsHelper2(matrix, start, dest, visited):
    if matrix[start[0]][start[1]] == 1:
        return False
    
    if visited[start[0]][start[1]]:
        return False
    if start[0] == dest[0] and start[1] == dest[1]:
        return True
    
    visited[start[0]][start[1]] = True
    
    r = start[1] + 1
    l = start[1] - 1
    u = start[0] - 1
    d = start[0] + 1
    
    while (r < len(matrix[0]) and matrix[start[0]][r] == 0):  ##  right
        r += 1
    x = (start[0], r - 1)
    if (dfsHelper2(matrix, x, dest, visited)):
        return True

    while (l >= 0 and matrix[start[0]][l] == 0):  ##  left
        l -= 1
    x = (start[0], l + 1)
    if (dfsHelper2(matrix, x, dest, visited)):
        return True
    
    while (u >= 0 and matrix[u][start[1]] == 0): ##  up
        u -= 1
    x = (u + 1, start[1])
    if (dfsHelper2(matrix, x, dest, visited)):
        return True
        
    while (d < len(matrix) and matrix[d][start[1]] == 0): ##  down
        d += 1
    x = (d - 1, start[1])
    if (dfsHelper2(matrix, x, dest, visited)):
        return True
            
    return False

if __name__ == '__main__':
	matrix = [
	    [0, 0, 1, 0, 0],
	    [0, 0, 0, 0, 0],
	    [0, 0, 0, 1, 0],
	    [1, 1, 0, 1, 1],
	    [0, 0, 0, 0, 0]
	]

	start = (0, 0)
	dest  = (3, 2)
	print(dfs2(matrix, start, dest))

# 3,迷宫(3):
import heapq

def shortestDistance(matrix, start, destination):
    def neighbors(matrix, node):
        for dir in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
            cur_node, dist = list(node), 0
            while 0 <= cur_node[0] + dir[0] < len(matrix) and \
                  0 <= cur_node[1] + dir[1] < len(matrix[0]) and \
                  matrix[cur_node[0] + dir[0]][cur_node[1] + dir[1]] == 0:
                cur_node[0] += dir[0]
                cur_node[1] += dir[1]
                dist += 1
            yield dist, tuple(cur_node)

    heap = [(0, start)]
    visited = set()
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited: continue
        if node == destination:
            return dist
        visited.add(node)
        for neighbor_dist, neighbor in neighbors(matrix, node):
            heapq.heappush(heap, (dist + neighbor_dist, neighbor))

    return -1
if __name__ == '__main__':
	matrix = [
	    [0, 0, 1, 0, 0],
	    [0, 0, 0, 0, 0],
	    [0, 0, 0, 1, 0],
	    [1, 1, 0, 1, 1],
	    [0, 0, 0, 0, 0]
	]

	start = (0, 4)
	dest  = (4, 4)
	shortestDistance(matrix, start, dest)

# 4,迷宫(4):
import heapq

def findShortestWay(maze, ball, hole):
    dirs = {'u' : (-1, 0), 'r' : (0, 1), 'l' : (0, -1), 'd': (1, 0)}

    def neighbors(maze, node):
        for dir, vec in dirs.items():
            cur_node, dist = list(node), 0
            while 0 <= cur_node[0]+vec[0] < len(maze) and \
                  0 <= cur_node[1]+vec[1] < len(maze[0]) and \
                  not maze[cur_node[0]+vec[0]][cur_node[1]+vec[1]]:
                cur_node[0] += vec[0]
                cur_node[1] += vec[1]
                dist += 1
                if tuple(cur_node) == hole:
                    break
            yield tuple(cur_node), dir, dist

    heap = [(0, '', ball)]
    visited = set()
    while heap:
        dist, path, node = heapq.heappop(heap)
        if node in visited: continue
        if node == hole: return path
        visited.add(node)
        for neighbor, dir, neighbor_dist in neighbors(maze, node):
            heapq.heappush(heap, (dist+neighbor_dist, path+dir, neighbor))

    return "impossible"

if __name__ == '__main__':
	matrix = [
	    [0, 0, 1, 0, 0],
	    [0, 0, 0, 0, 0],
	    [0, 0, 0, 1, 0],
	    [1, 1, 0, 1, 1],
	    [0, 0, 0, 0, 0]
	]

	start = (0, 0)
	dest  = (1, 4)
	findShortestWay(matrix, start, dest)

# 5,Flood Fill:
def floodFill(image, sr, sc, newColor):
    rows, cols, orig_color = len(image), len(image[0]), image[sr][sc]
    def traverse(row, col):
        if (not (0 <= row < rows and 0 <= col < cols)) or image[row][col] != orig_color:
            return
        image[row][col] = newColor
        [traverse(row + x, col + y) for (x, y) in ((0, 1), (1, 0), (0, -1), (-1, 0))]
    if orig_color != newColor:
        traverse(sr, sc)
    return image

if __name__ == '__main__':
	image = [
	    [1,1,1],
	    [1,1,0],
	    [1,0,1]
	]
	sr = 1
	sc = 1
	newColor = 2
	floodFill(image, sr, sc, newColor)

# 6,Friend Circles:
def findCircleNum(M):
    circle = 0
    n = len(M)
    for i in range(n):
        if M[i][i] != 1:
            continue
        friends = [i]
        while friends:
            f = friends.pop()
            if M[f][f] == 0:
                continue
            M[f][f] = 0
            for j in range(n):
                if M[f][j] == 1 and M[j][j] == 1:
                    friends.append(j)
        circle += 1
    return circle

def findCircleNum2(M):
    def dfs(node):
        visited.add(node)
        for friend in range(len(M)):
            if M[node][friend] and friend not in visited:
                dfs(friend)

    circle = 0
    visited = set()
    for node in range(len(M)):
        if node not in visited:
            dfs(node)
            circle += 1
    return circle

if __name__ == '__main__':
	M = [
	     [1,1,0],
	     [1,1,0],
	     [0,0,1]]
	print(findCircleNum(M))

	print(M)
	print(findCircleNum2(M))

# 7,Number of Islands:
def numIslands(grid):
    if not grid:
        return 0
    count = 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                dfs(grid, i, j)
                count += 1
    return count

def dfs(grid, i, j):
    if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != 1:
        return
    grid[i][j] = '#'
    dfs(grid, i + 1, j)
    dfs(grid, i - 1, j)
    dfs(grid, i, j + 1)
    dfs(grid, i, j - 1)

if __name__ == '__main__':
	M = [
	    [1,1,0,0,0],
	    [1,1,0,0,0],
	    [0,0,1,0,0],
	    [0,0,0,1,1]
	]
	numIslands(M)

# 8,Max Area of Island:
def maxAreaOfIsland(grid):
    m, n = len(grid), len(grid[0])

    def dfs(i, j):
        if 0 <= i < m and 0 <= j < n and grid[i][j]:
            grid[i][j] = 0
            return 1 + dfs(i - 1, j) + dfs(i, j + 1) + dfs(i + 1, j) + dfs(i, j - 1)
        return 0

    result = 0
    for x in range(m):
        for y in range(n):
            if grid[x][y]:
                result = max(result, dfs(x, y))
    return result

if __name__ == '__main__':
	matrix = [
	    [0, 0, 1, 0, 0],
	    [0, 0, 0, 0, 0],
	    [0, 0, 0, 1, 0],
	    [1, 1, 0, 1, 1],
	    [0, 0, 0, 0, 0]
	]

	maxAreaOfIsland(matrix)

# 9,Employee Importance:
class Employee(object):
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates

def getImportance(employees, id):
    table = {emp.id: emp for emp in employees}

    def dfs(emp):
        if emp.subordinates == []:  # base case
            return emp.importance
        else:  # recursive case
            value = emp.importance
            for sub in emp.subordinates:
                value += dfs(table[sub])
            return value
            # or just:
            # return emp.importance + sum(dfs(table[sub]) for sub in emp.subordinates)

    return dfs(table[id])

def getImportance2(employees, id):
    value = 0
    table = {}
    for emp in employees:
        table[emp.id] = emp

    stack = [table[id]]

    while stack:
        emp = stack.pop()
        for sub in emp.subordinates:
            stack.append(table[sub])
        value += emp.importance

    return value

if __name__ == '__main__':
	e3 = Employee(3, 5, [])
	e2 = Employee(2, 10, [3])
	e1 = Employee(1, 15, [2])
	emps = [e1, e2, e3]
	getImportance2(emps, 1)

# 10, Is Graph Bipartite：
def isBipartite(graph):
    color = {}
    def dfs(pos):
        for i in graph[pos]:
            if i in color:
                if color[i] == color[pos]: return False
            else:
                color[i] = color[pos] ^ 1
                if not dfs(i): return False
        return True
    
    for i in range(len(graph)):
        if i not in color: color[i] = 0
        if not dfs(i): return False
    return True

if __name__ == '__main__':
	graph = [[1,3], [0,2], [1,3], [0,2]]
	print(isBipartite(graph))

# 11,Pacific Atlantic Water Flow:
def pacificAtlantic(matrix):

    if not matrix: return []
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    m = len(matrix)
    n = len(matrix[0])
    p_visited = [[False for _ in range(n)] for _ in range(m)]

    a_visited = [[False for _ in range(n)] for _ in range(m)]
    result = []

    for i in range(m):
        # p_visited[i][0] = True
        # a_visited[i][n-1] = True
        dfs(matrix, i, 0, p_visited, m, n)
        dfs(matrix, i, n-1, a_visited, m, n)
    for j in range(n):
        # p_visited[0][j] = True
        # a_visited[m-1][j] = True
        dfs(matrix, 0, j, p_visited, m, n)
        dfs(matrix, m-1, j, a_visited, m, n)

    for i in range(m):
        for j in range(n):
            if p_visited[i][j] and a_visited[i][j]:
                result.append([i,j])
    return result


def dfs(matrix, i, j, visited, m, n):
    # when dfs called, meaning its caller already verified this point 
    visited[i][j] = True
    for dir in [(1,0),(-1,0),(0,1),(0,-1)]:
        x, y = i + dir[0], j + dir[1]
        if x < 0 or x >= m or y < 0 or y >= n or visited[x][y] or matrix[x][y] < matrix[i][j]:
            continue
        dfs(matrix, x, y, visited, m, n)

from collections import deque

def pacificAtlantic2(matrix):
    if not matrix: return []
    m, n = len(matrix), len(matrix[0])
    def bfs(reachable_ocean):
        q = deque(reachable_ocean)
        while q:
            (i, j) = q.popleft()
            for (di, dj) in [(0,1), (0, -1), (1, 0), (-1, 0)]:
                if 0 <= di+i < m and 0 <= dj+j < n and (di+i, dj+j) not in reachable_ocean \
                    and matrix[di+i][dj+j] >= matrix[i][j]:
                    q.append( (di+i,dj+j) )
                    reachable_ocean.add( (di+i, dj+j) )
        return reachable_ocean         
    pacific  =set ( [ (i, 0) for i in range(m)]   + [(0, j) for j  in range(1, n)]) 
    atlantic =set ( [ (i, n-1) for i in range(m)] + [(m-1, j) for j in range(n-1)]) 
    return list( bfs(pacific) & bfs(atlantic) )

if __name__ == '__main__':
	matrix = [
	    [1,2,2,3,4],
	    [3,2,3,4,4],
	    [2,4,5,3,1],
	    [6,7,1,4,5],
	    [5,1,1,2,4]
	]
	print(pacificAtlantic2(matrix))

# 12,Longest Increasing Path in a Matrix:
def longestIncreasingPath(matrix):
    if not matrix: return 0
    directions = [(1,0),(-1,0),(0,1),(0,-1)]
    m = len(matrix)
    n = len(matrix[0])
    cache = [[-1 for _ in range(n)] for _ in range(m)]
    res = 0
    for i in range(m):
        for j in range(n):
            cur_len = dfs(i, j, matrix, cache, m, n)
            res = max(res, cur_len)
    return res

def dfs(i, j, matrix, cache, m, n):
    if cache[i][j] != -1:
        return cache[i][j]
    res = 1
    for direction in [(1,0),(-1,0),(0,1),(0,-1)]:
        x, y = i + direction[0], j + direction[1]
        if x < 0 or x >= m or y < 0 or y >= n or matrix[x][y] <= matrix[i][j]:
            continue
        length = 1 + dfs(x, y, matrix, cache, m, n)
        res = max(length, res)
    cache[i][j] = res
    return res

if __name__ == '__main__':
	nums = [
	  [8,4,5],
	  [3,9,6],
	  [2,8,7]
	]
	longestIncreasingPath(nums)

# 13,Matrix:
def updateMatrix(matrix):
    q, m, n = [], len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(n):
            if matrix[i][j] != 0:
                matrix[i][j] = 0x7fffffff
            else:
                q.append((i, j))
    for i, j in q:
        for r, c in ((i, 1+j), (i, j-1), (i+1, j), (i-1, j)):
            z = matrix[i][j] + 1
            if 0 <= r < m and 0 <= c < n and matrix[r][c] > z:
                matrix[r][c] = z
                q.append((r, c))
    return matrix

def updateMatrix2(matrix):
    def DP(i, j, m, n, dp):
        if i > 0: dp[i][j] = min(dp[i][j], dp[i - 1][j] + 1)
        if j > 0: dp[i][j] = min(dp[i][j], dp[i][j - 1] + 1)
        if i < m - 1: dp[i][j] = min(dp[i][j], dp[i + 1][j] + 1)
        if j < n - 1: dp[i][j] = min(dp[i][j], dp[i][j + 1] + 1)
            
    if not matrix: return [[]]
    m, n = len(matrix), len(matrix[0])
    dp = [[0x7fffffff if matrix[i][j] != 0 else 0 for j in range(n)] for i in range(m)]
    for i in range(m):
        for j in range(n):
            DP(i, j, m, n, dp)

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            DP(i, j, m, n, dp)

    return dp

if __name__ == '__main__':
	matrix = [
	    [0, 0, 0],
	    [0, 1, 0],
	    [0, 0, 0],
	]
	print(updateMatrix2(matrix))

# 14,Accounts Merge:
def accountsMerge(accounts):
    from collections import defaultdict
    visited_accounts = [False] * len(accounts)
    emails_accounts_map = defaultdict(list)
    res = []
    # Build up the graph.
    for i, account in enumerate(accounts):
        for j in range(1, len(account)): #email starts from 2nd
            email = account[j]
            emails_accounts_map[email].append(i)
            
    print(emails_accounts_map)
    # DFS code for traversing accounts.
    def dfs(i, emails):
        if visited_accounts[i]:
            return
        visited_accounts[i] = True
        for j in range(1, len(accounts[i])):
            email = accounts[i][j]
            emails.add(email)
            for neighbor in emails_accounts_map[email]:
                dfs(neighbor, emails)
    # Perform DFS for accounts and add to results.
    for i, account in enumerate(accounts):
        if visited_accounts[i]:
            continue
        name, emails = account[0], set()
        dfs(i, emails)
        res.append([name] + sorted(emails))
    return res

if __name__ == '__main__':
	accounts = [
	    ["John", "johnsmith@mail.com", "john00@mail.com"], 
	    ["John", "johnnybravo@mail.com"], 
	    ["John", "johnsmith@mail.com", "john_newyork@mail.com"], 
	    ["Mary", "mary@mail.com"]
	]

	print(accountsMerge(accounts))

# 15,Word Ladder:
from collections import deque
def ladderLength(beginWord, endWord, wordList):
    wordSet=set(wordList)
    wordSet.add(endWord)
    queue = deque([[beginWord, 1]])
    while queue:
        word, length = queue.popleft()
        if word == endWord:
            return length
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in wordSet:
                    wordSet.remove(next_word)
                    queue.append([next_word, length + 1])
    return 0

if __name__ == '__main__':
	beginWord = "hit"
	endWord = "cog"
	wordList = ["hot","dot","dog","lot","log","cog"]

	ladderLength(beginWord, endWord, wordList)


# 16,Word Ladder(2):
from collections import defaultdict
import string
def findLadders(start, end, wordList):
    dic = set(wordList)
    dic.add(end)
    level = {start}
    parents = defaultdict(set)
    while level and end not in parents:
        next_level = defaultdict(set)
        for node in level:
            for char in string.ascii_lowercase:
                for i in range(len(start)):
                    n = node[:i] + char + node[i+1:]
                    if n in dic and n not in parents:
                        next_level[n].add(node)
        level = next_level
        parents.update(next_level)
    res = [[end]]
    print(parents)
    while res and res[0][0] != start:
        res = [[p]+r for r in res for p in parents[r[0]]]
    return res

if __name__ == '__main__':
	beginWord = "hit"
	endWord = "cog"
	wordList = ["hot","dot","dog","lot","log","cog"]
	findLadders(beginWord, endWord, wordList)

# 17,topologicalSort：
def topologicalSort(G):
    """Perform a topological sort of the nodes. If the graph has a cycle,
    throw a GraphTopologicalException with the list of successfully
    ordered nodes."""
    # topologically sorted list of the nodes (result)
    topologicalList = []
    # queue (fifo list) of the nodes with inDegree 0
    topologicalQueue = []
    # {node: inDegree} for the remaining nodes (those with inDegree>0)
    remainingInDegree = {}

    nodes = G.getVertices()
    for v in G:
        indegree = v.getInDegree()
        if indegree == 0:
            topologicalQueue.append(v)
        else:
            remainingInDegree[v] = indegree

    # remove nodes with inDegree 0 and decrease the inDegree of their sons
    while len(topologicalQueue):
        # remove the first node with degree 0
        node = topologicalQueue.pop(0)
        topologicalList.append(node)
        # decrease the inDegree of the sons
        for son in node.getConnections():
            son.setInDegree(son.getInDegree() - 1)
            if son.getInDegree() == 0:
                topologicalQueue.append(son)

    # if not all nodes were covered, the graph must have a cycle
    # raise a GraphTopographicalException
    if len(topologicalList) != len(nodes):
        raise ValueError(topologicalList)

    # Printing the topological order    
    while len(topologicalList):
        node = topologicalList.pop(0)
        print(node.getVertexID())

# if __name__ == '__main__':
	# G = Graph(True)
	# G.addVertex('A')
	# G.addVertex('B')
	# G.addVertex('C')
	# G.addVertex('D')
	# G.addVertex('E')
	# G.addVertex('F')
	# G.addVertex('G')
	# G.addVertex('H')
	# G.addVertex('I')
	# G.addEdge('A', 'B')   
	# G.addEdge('A', 'C')   
	# G.addEdge('A', 'G')  
	# G.addEdge('A', 'E')  
	# G.addEdge('B', 'C')       
	# G.addEdge('C', 'G')   
	# G.addEdge('D', 'E')  
	# G.addEdge('D', 'F')  
	# G.addEdge('F', 'H')       
	# G.addEdge('E', 'H')    
	# G.addEdge('H', 'I')      
	# print('Graph data:')
	# print(G.getEdges())   
	# topologicalSort(G)