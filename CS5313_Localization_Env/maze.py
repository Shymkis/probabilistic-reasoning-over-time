"""Random Maze Generator using Depth-first Search\n
http://en.wikipedia.org/wiki/Maze_generation_algorithm\n
FB - 2012-12-14\n

Code taken from http://code.activestate.com/recipes/578356-random-maze-generator/ used and modified under MIT license\n
Edits by Robert Geraghty\n
"""
import random


def make_maze(x, y, seed):
    random.seed(seed)
    mx = x; my = y # width and height of the maze
    maze = [[1 for y in range(my)] for x in range(mx)]
    dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
    # start the maze from a random cell

    stack = [(random.randint(1, mx - 2), random.randint(1, my - 2))]
    while len(stack) > 0:
        (cx, cy) = stack[-1]
        maze[cx][cy] = 0
        # find a new cell to add
        nlst = [] # list of available neighbors
        for i in range(4):
            nx = cx + dx[i]; ny = cy + dy[i]
            if nx >= 1 and nx < mx-1 and ny >= 1 and ny < my-1:
                if maze[nx][ny] == 1:
                    # of occupied neighbors must be 1
                    ctr = 0
                    for j in range(4):
                        ex = nx + dx[j]; ey = ny + dy[j]
                        if ex >= 0 and ex < mx and ey >= 0 and ey < my:
                            if maze[ex][ey] == 0: ctr += 1
                    if ctr == 1: nlst.append(i)
        # if 1 or more neighbors available then randomly select one and move
        if len(nlst) > 0:
            ir = nlst[random.randint(0, len(nlst) - 1)]
            cx += dx[ir]; cy += dy[ir]
            stack.append((cx, cy))
        else: stack.pop()
    return maze
