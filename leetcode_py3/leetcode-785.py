# leetcode-785：二分图：beats 80.75%
class Solution:
    def isBipartite(self, graph: List[List[int]]) -> bool:
        colors = [-1] * len(graph)
        for i in range(len(graph)):
            if colors[i] == -1 and not self.dfs(i, 0, colors, graph):
                return False
        return True
    
    def dfs(self, cur_node, cur_color, colors, graph):
        if colors[cur_node] != -1:
            return colors[cur_node] == cur_color
        
        colors[cur_node] = cur_color
        for next_node in graph[cur_node]:
            if not self.dfs(next_node, 1-cur_color, colors, graph):
                return False
        return True