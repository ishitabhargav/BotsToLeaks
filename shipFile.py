import numpy as np
import heapq


def getValidNeighbors(row, col, size) -> list[(int, int)]:  # neighbors that are inbounds
    validNeighbors = []
    if row + 1 < size:
        validNeighbors.append((row + 1, col))
    if col + 1 < size:
        validNeighbors.append((row, col + 1))
    if row - 1 >= 0:
        validNeighbors.append((row - 1, col))
    if col - 1 >= 0:
        validNeighbors.append((row, col - 1))
    return validNeighbors


def numOpenNeighbors(validNeighbors, arr) -> int:
    count = 0
    for neighbor in validNeighbors:
        if arr[neighbor[0]][neighbor[1]][0] == 1: # count the number of open neighbors
            count = count + 1
    return count


def addToClosedNeighbors(row, col, arr, closedNeighbors, size):  # adding inbound, closed, and not in closedNeighbors list neighbors to closedNeighbors
    validNeighbors = getValidNeighbors(row, col, size)
    for neighbor in validNeighbors:
        if arr[neighbor[0]][neighbor[1]][0] == 0 and neighbor not in closedNeighbors and numOpenNeighbors(
                getValidNeighbors(neighbor[0], neighbor[1], size), arr) == 1:
            closedNeighbors.append(neighbor)


def findDistanceBetween(x1, y1, x2, y2, size, arr) -> int:
    if x1 == x2 and y1 == y2 and arr[x1][y1][0] != 0 and arr[x1][y1][0] != 2:
        return 0
    queue = [(x1, y1)]
    distanceFromX1Y1 = {(x1, y1): 0}
    visited = np.zeros((size, size))
    visited[x1][y1] = 1
    while queue:
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1 or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0 and arr[next_x][next_y][0] != 2:
                if next_x == x2 and next_y == y2:
                    return distanceFromX1Y1[(x, y)] + 1
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromX1Y1[(next_x, next_y)] = distanceFromX1Y1[(x, y)] + 1
    return -1  # path not found


def findDistanceBetweenBot3(x1, y1, x2, y2, size, arr) -> int:
    if x1 == x2 and y1 == y2 and arr[x1][y1][0] != 0 and arr[x1][y1][0] != 2:
        return 0
    queue = [(x1, y1)]
    distanceFromX1Y1 = {(x1, y1): 0}
    visited = np.zeros((size, size))
    visited[x1][y1] = 1
    while queue:
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1 or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0 and arr[next_x][next_y][0] != 2 and \
                    arr[next_x][next_y][0] != 3:
                if next_x == x2 and next_y == y2:
                    return distanceFromX1Y1[(x, y)] + 1
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromX1Y1[(next_x, next_y)] = distanceFromX1Y1[(x, y)] + 1
    return -1  # path not found


def findDistanceBetweenBot4(x1, y1, x2, y2, size, arr, q, cost1F, cost2F, benefit1, benefit2, benefit3) -> int:
    if x1 == x2 and y1 == y2 and arr[x1][y1][0] != 0 and arr[x1][y1][0] != 2:
        return 0
    pq = []
    heapq.heappush(pq, (0, (x1, y1)))
    visited = np.zeros((size, size))
    visited[x1][y1] = 1
    while pq:
        priority, item = heapq.heappop(pq)
        x = item[0]
        y = item[1]
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1 or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0 and arr[next_x][next_y][
                0] != 2:  # and alreadyVisited[next_x][next_y] == 0:
                if next_x == x2 and next_y == y2:
                    return priority + 1
                visited[next_x][next_y] = 1
                next_priority = priority + 1
                if arr[next_x][next_y][0] == cost1F or arr[next_x][next_y][0] == cost2F:
                    next_priority = next_priority + arr[next_x][next_y][0] * q
                    if arr[next_x][next_y][1] == benefit1 or arr[next_x][next_y][1] == benefit2 or arr[next_x][next_y][
                        1] == benefit3:
                        next_priority = next_priority - arr[next_x][next_y][1] * q
                heapq.heappush(pq, (next_priority, (next_x, next_y)))
    return -1  # path not found


def findDistanceBetweenBot1(x1, y1, x2, y2, size, arr, firstCellOnFire) -> int:
    if x1 == x2 and y1 == y2 and arr[x1][y1][0] != 0 and (x1, x2) != firstCellOnFire:
        return 0
    queue = [(x1, y1)]
    distanceFromX1Y1 = {(x1, y1): 0}
    visited = np.zeros((size, size))
    visited[x1][y1] = 1
    while queue:
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for any cell as long as it's not the initial fire cell, firstCellOnFire
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0 and (next_x, next_y) != firstCellOnFire:
                if next_x == x2 and next_y == y2:
                    return distanceFromX1Y1[(x, y)] + 1
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromX1Y1[(next_x, next_y)] = distanceFromX1Y1[(x, y)] + 1
    return -1  # path not found


class Ship:
    def __init__(self):
        self.size = 30
        self.arr = np.zeros((self.size, self.size, 2))
        self.randRow = np.random.randint(1, self.size - 1)
        self.randCol = np.random.randint(1, self.size - 1)
        self.arr[self.randRow][self.randCol][0] = 1  # open first cell
        self.closedNeighbors = []
        self.openCells = [[self.randRow, self.randCol]]
        addNeighbors(self.randRow, self.randCol, self.arr, self.closedNeighbors, self.size)

        while self.closedNeighbors:  # choose a closed neighbor at random to open, from list of closed cells with one open neighbor
            # 1. pick a closed neighbor and open it at random
            self.sizeClosedNeighbors = len(self.closedNeighbors)
            self.rand = np.random.randint(0, self.sizeClosedNeighbors)
            self.neighborToOpen = self.closedNeighbors.pop(self.rand)
            self.rowToOpen = self.neighborToOpen[0]
            self.colToOpen = self.neighborToOpen[1]
            self.arr[self.rowToOpen][self.colToOpen][0] = 1
            self.openCells.append([self.rowToOpen, self.colToOpen])
            # 2. remove existing neighbors in closedNeighbors that now have 2 or more open neighbors
            self.validNeighbors = getValidNeighbors(self.rowToOpen, self.colToOpen, self.size)
            for neighbor in self.validNeighbors:
                if self.arr[neighbor[0]][neighbor[1]][0] == 0:  # it's a closed neighbor
                    self.neighborsOfClosed = getValidNeighbors(neighbor[0], neighbor[1], self.size)
                    if numOpenNeighbors(self.neighborsOfClosed, self.arr) > 1 and neighbor in self.closedNeighbors:
                        self.closedNeighbors.remove(neighbor)
            # 3. add the closed neighbors of neighborToOpen
            addToClosedNeighbors(self.rowToOpen, self.colToOpen, self.arr, self.closedNeighbors, self.size)

        self.deadEnds = []
        for cell in self.openCells:
            if numOpenNeighbors(getValidNeighbors(cell[0], cell[1], self.size), self.arr) == 1:
                self.deadEnds.append(cell)

        self.origNumDeadEnds = len(self.deadEnds)

        while len(self.deadEnds) > 0.50 * self.origNumDeadEnds:
            randCell = np.random.randint(0, len(self.deadEnds))
            deadEnd = self.deadEnds.pop(randCell)
            deadEndsClosedNeighbors = []
            validNeighbors = getValidNeighbors(deadEnd[0], deadEnd[1], self.size)
            for neighbor in validNeighbors:  # can change to x, y
                if self.arr[neighbor[0]][neighbor[1]][0] == 0:
                    deadEndsClosedNeighbors.append(neighbor)
            if not deadEndsClosedNeighbors:
                randCell2 = np.random.randint(0, len(deadEndsClosedNeighbors))
                deadEndNeighbor = deadEndsClosedNeighbors.pop(randCell2)
                self.arr[deadEndNeighbor[0]][deadEndNeighbor[1]][0] = 1
                for deadEnd in self.deadEnds:
                    if numOpenNeighbors(getValidNeighbors(deadEnd[0], deadEnd[1], self.size), self.arr) != 1:
                        self.deadEnds.remove(deadEnd)
                self.openCells.append(deadEndNeighbor)

        # pick 3 different locations for fire, bot, and button
        self.indexFirstCellOnFire = np.random.randint(0, len(self.openCells))
        self.firstCellOnFire = self.openCells.pop(self.indexFirstCellOnFire)
        self.arr[self.firstCellOnFire[0]][self.firstCellOnFire[1]][0] = 2  # this cell is on fire
        self.indexBot = np.random.randint(0, len(self.openCells))
        self.bot = self.openCells.pop(self.indexBot)
        self.indexButton = np.random.randint(0, len(self.openCells))
        self.button = self.openCells.pop(self.indexButton)

        # add the cells back into the openCells list
        self.openCells.append(self.firstCellOnFire)
        self.openCells.append(self.bot)
        self.openCells.append(self.button)

        '''for row in self.arr:
            for col in row:
                if col[0] == 1:  # it's open
                    print("1", end="")
                elif col[0] == 0:  # it's closed
                    print("O", end="")
                else:  # it's on fire
                    print("2", end="")
            print("\n")'''
