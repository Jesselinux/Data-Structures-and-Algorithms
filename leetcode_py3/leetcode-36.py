class Solution:
    def isValidSudoku(self, board):
        raw = [{},{},{},{},{},{},{},{},{}]
        col = [{},{},{},{},{},{},{},{},{}]
        cell = [{},{},{},{},{},{},{},{},{}]
        for i in range(9):
            for j in range(9):                                 
                num = (3*(i//3) + j//3)#找单元
                temp = board[i][j]
                if temp != ".":
                    if temp not in raw[i] and temp not in col[j] and temp not in cell[num]:
                        raw [i][temp] = 1
                        col [j][temp] = 1
                        cell [num][temp] =1
                    else:
                        return False    
        return True