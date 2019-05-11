# leetcode-207：拓扑排序，beats 90.89%
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        graphic = [[] for i in range(numCourses)]
        for pre in prerequisites:
            graphic[pre[0]].append(pre[1])
        visited = [0]*numCourses
        for i in range(numCourses):
            if self.exist_cycle(visited, graphic, i):
                return False
        return True
    
    def exist_cycle(self, visited, graphic, cur_node):
        if visited[cur_node] == 1:
            return True
        if visited[cur_node] == 2:
            return False
        visited[cur_node] = 1
        for next_node in graphic[cur_node]:
            if self.exist_cycle(visited, graphic, next_node):
                return True
        visited[cur_node] = 2
        return False