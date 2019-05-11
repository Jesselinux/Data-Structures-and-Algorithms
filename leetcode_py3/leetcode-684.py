# leetcode-684：并查集，beats  5.42%
class UnionFind:
    def __init__(self, n):
        self.ids = []
        for i in range(n+1):
            self.ids.append(i)
            
    def union(self, u, v):
        u_id = self.find(u)
        v_id = self.find(v)
        if u_id == v_id:
            return
        for i in range(len(self.ids)):
            if self.ids[i] == u_id:
                self.ids[i] = v_id
    
    def find(self, p):
        return self.ids[p]
    
    def connect(self, u, v):
        return self.find(u) == self.find(v)
                
    
class Solution:    
    def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
        uf = UnionFind(len(edges))
        for e in edges:
            u, v = e[0], e[1]
            if uf.connect(u, v):
                return u, v
            uf.union(u, v)
        return -1, -1