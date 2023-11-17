from math import exp
from typing import Set, Any
import math
import numpy as np
from ship_file import Ship, getValidNeighbors
import matplotlib.pyplot as plt
import random


# FUNCTIONS HERE

def getInBoundsVal(value, size) -> int:
    if value < 0:
        return 0
    if value >= size:
        return size - 1
    return value


def getOpenSquareCells(arr, k, bot, size) -> list[(int, int)]:
    openCellsInSquare = []
    botX, botY = bot
    topX = getInBoundsVal(botX - k, size)
    bottomX = getInBoundsVal(botX + k, size)
    rightY = getInBoundsVal(botY + k, size)
    leftY = getInBoundsVal(botY - k, size)

    for row in range(topX, bottomX + 1):
        for col in range(leftY, rightY + 1):
            if arr[row][col][0] == 1:  # open, unexplored cell
                openCellsInSquare.append((row, col))
    return openCellsInSquare


def getOpenCellsAndCheckLeak(arr, k, bot, size) -> list[list[(int, int)], bool]:
    openCellsInSquare = []
    leakDetected = False
    botX, botY = bot
    topX = getInBoundsVal(botX - k, size)
    bottomX = getInBoundsVal(botX + k, size)
    rightY = getInBoundsVal(botY + k, size)
    leftY = getInBoundsVal(botY - k, size)

    for row in range(topX, bottomX + 1):
        for col in range(leftY, rightY + 1):
            if arr[row][col][0] == 1:  # open, unexplored cell
                openCellsInSquare.append((row, col))
            elif arr[row][col][0] == 3:  # leak cell!
                openCellsInSquare.append((row, col))
                leakDetected = True
    return [openCellsInSquare, leakDetected]


def findClosestUnexploredNeighbors(botRow, botCol, size, arr) -> list[list[(int, int)], int]:
    unexploredNeighbors = []
    queue = [(botRow, botCol)]
    distanceFromBot = {(botRow, botCol): 0}
    visited = np.zeros((size, size))
    visited[botRow][botCol] = 1
    currDistance = 0
    distOfClosestUnexplored = 0  # represents distance of neighbors that will be returned...maybe rename variable
    noNeighborsYet = True
    while queue and (not unexploredNeighbors or (unexploredNeighbors and currDistance == distOfClosestUnexplored)):
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1 or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0:
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromBot[(next_x, next_y)] = distanceFromBot[(x, y)] + 1
                currDistance = distanceFromBot[(next_x, next_y)]
                if arr[next_x][next_y][0] != 2:
                    if noNeighborsYet:
                        noNeighborsYet = False
                        distOfClosestUnexplored = currDistance
                    elif currDistance > distOfClosestUnexplored:
                        break
                    unexploredNeighbors.append((next_x, next_y))
    return [unexploredNeighbors, distOfClosestUnexplored]


def findUnexploredNeighborsInSquare(botRow, botCol, size, k, numTimesLeakDetected, arr) -> list[
    list[(int, int)], int]:
    if k > 14:
        goalDist = (30 - k) // (4 * numTimesLeakDetected)
    else:
        goalDist = math.ceil(k / (6 * numTimesLeakDetected))
    unexploredNeighbors = []
    farthestUnexploreNeighborsInSquare = []
    queue = [(botRow, botCol)]
    distanceFromBot = {(botRow, botCol): 0}
    visited = np.zeros((size, size))
    visited[botRow][botCol] = 1
    currDistance = 0
    noNeighborsYet = True
    while queue and (currDistance <= goalDist or currDistance == firstUnexploredDist):
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            if next_x > botRow + goalDist or next_x < botRow - goalDist or next_y > botCol + goalDist or next_y < botCol - goalDist:
                continue
            # calculate distance for cell of val 1, 2, or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0:
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromBot[(next_x, next_y)] = distanceFromBot[(x, y)] + 1
                currDistance = distanceFromBot[(next_x, next_y)]
                if noNeighborsYet:
                    firstUnexploredDist = currDistance
                if arr[next_x][next_y][0] != 2 and (currDistance <= goalDist or currDistance == firstUnexploredDist):
                    if noNeighborsYet:
                        noNeighborsYet = False
                    unexploredNeighbors.append((next_x, next_y))
                elif currDistance > goalDist and currDistance > firstUnexploredDist:
                    break
    if not unexploredNeighbors:
        return findClosestUnexploredNeighbors(botRow, botCol, size, arr)
    farthestUnexploredNeighborsInSquareDist = distanceFromBot[unexploredNeighbors[-1]]
    for cell in unexploredNeighbors:
        if distanceFromBot[(cell[0], cell[1])] == farthestUnexploredNeighborsInSquareDist:
            farthestUnexploreNeighborsInSquare.append(cell)
    return [farthestUnexploreNeighborsInSquare, farthestUnexploredNeighborsInSquareDist]


# returns unexplored neighbors of distance 2k or distance closest to 2k (> 2k is possible, otherwise <2k)
def findSomeDistUnexploredNeighbors(botRow, botCol, size, k, numTimesLeakDetected, arr) -> list[list[(int, int)], int]:
    if numTimesLeakDetected > 0:
        goalDist = k + 1
    else:
        goalDist = (8 * k) // 7
    unexploredNeighbors = []
    twoKDistUnexploredNeighbors = []
    queue = [(botRow, botCol)]
    distanceFromBot = {(botRow, botCol): 0}
    visited = np.zeros((size, size))
    visited[botRow][botCol] = 1
    currDistance = 0
    noNeighborsYet = True
    firstunexploredDist = 0
    while queue and (not unexploredNeighbors or (
            unexploredNeighbors and (currDistance <= goalDist or currDistance == firstunexploredDist))):
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1, 2, or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0:
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromBot[(next_x, next_y)] = distanceFromBot[(x, y)] + 1
                currDistance = distanceFromBot[(next_x, next_y)]
                if noNeighborsYet:
                    firstunexploredDist = currDistance
                if arr[next_x][next_y][0] != 2 and (currDistance <= goalDist or currDistance == firstunexploredDist):
                    if noNeighborsYet:
                        noNeighborsYet = False
                    unexploredNeighbors.append((next_x, next_y))
                elif firstunexploredDist > goalDist and currDistance > firstunexploredDist:
                    break
    closestToKUnexploredDist = distanceFromBot[unexploredNeighbors[-1]]
    for cell in unexploredNeighbors:
        if distanceFromBot[(cell[0], cell[1])] == closestToKUnexploredDist:
            twoKDistUnexploredNeighbors.append(cell)
    return [twoKDistUnexploredNeighbors, closestToKUnexploredDist]


def getAllUnexploredOutsideSquare(arr, size, openCellsInSquare) -> list[(int, int)]:
    allUnexploredCells = []
    # calculate all open unexplored cells
    for row in range(0, size):
        for col in range(0, size):
            if arr[row][col][0] == 1 or arr[row][col][0] == 3:
                allUnexploredCells.append((row, col))

    # if any of these cells are in the detection square, remove them from allUnexploredCells
    for cell in openCellsInSquare:
        if cell in allUnexploredCells:
            allUnexploredCells.remove(cell)
    return allUnexploredCells


def getOpenCells(arr, size) -> list[(int, int)]:
    openCells = []
    for row in range(0, size):
        for col in range(0, size):
            if arr[row][col][0] == 1:
                openCells.append((row, col))
    return openCells


def printShip(arr, bot, size):
    for row in range(0, size):
        for col in range(0, size):
            if bot[0] == row and bot[1] == col:
                print("4", end="")  # it's the bot
            elif arr[row][col][0] == 0:  # it's closed
                print("0", end="")
            elif arr[row][col][0] == 1:  # it's unexplored
                print("1", end="")
            elif arr[row][col][0] == 2:  # it's explored and doesn't have the leak
                print("2", end="")
            else:
                print("3", end="")  # it's the leak
        print("\n")


def senseBeep(arr, bot, size, leak, alpha, distBotToLeak, prob) -> bool:
    rand = random.random()
    if rand < prob:
        return True
    return False


def findDistanceBetween(x1, y1, x2, y2, size, arr) -> int:
    if x1 == x2 and y1 == y2 and arr[x1][y1][0] != 0:
        return 0
    queue = [(x1, y1)]
    distanceFromX1Y1 = {(x1, y1): 0}
    visited = np.zeros((size, size))
    visited[x1][y1] = 1
    while queue:
        x, y = queue.pop(0)
        for next_x, next_y in getValidNeighbors(x, y, size):
            # calculate distance for cell of val 1 or 3
            if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0:
                if next_x == x2 and next_y == y2:
                    return distanceFromX1Y1[(x, y)] + 1
                queue.append((next_x, next_y))
                visited[next_x][next_y] = 1
                distanceFromX1Y1[(next_x, next_y)] = distanceFromX1Y1[(x, y)] + 1
    return -1  # path not found


def makePairsMatrix(pairsMatrix, openCells) -> np.ndarray[
    Any, np.dtype[Any]]:  # list[[list[(int, int)]]]: # set[Any]: #
    for i in range(0, len(openCells)):
        for j in range(i + 1, len(openCells)):
            one = openCells[i]
            two = openCells[j]
            pairsMatrix[one[0]][one[1]][two[0]][two[1]] = 1
            pairsMatrix[two[0]][two[1]][one[0]][one[1]] = 1
    return pairsMatrix


def eliminateCells(pairsMatrix, listToRemove, size):
    for item in listToRemove:
        x1 = item[0]
        y1 = item[1]
        for a in range(size):
            for b in range(size):
                pairsMatrix[a][b][x1][y1] = 0
                pairsMatrix[x1][y1][a][b] = 0


def eliminateOneCell(cell, pairsMatrix, size):
    x1 = cell[0]
    y1 = cell[1]
    for a in range(size):
        for b in range(size):
            pairsMatrix[a][b][x1][y1] = 0
            pairsMatrix[x1][y1][a][b] = 0


def BFS_traversal(arr, size, openCells):
    distancesHashtable = {}
    for cell in openCells:
        x1 = cell[0]
        y1 = cell[1]
        queue = [(x1, y1)]
        distanceFromX1Y1 = {(x1, y1): 0}
        visited = np.zeros((size, size))
        visited[x1][y1] = 1
        distancesHashtable[((x1, y1), (x1, y1))] = 0
        while queue:
            x, y = queue.pop(0)
            for next_x, next_y in getValidNeighbors(x, y, size):
                # calculate distance for cell of val 1 or 3
                if visited[next_x][next_y] != 1 and arr[next_x][next_y][0] != 0:
                    distancesHashtable[((x1, y1), (next_x, next_y))] = distanceFromX1Y1[(x, y)] + 1
                    queue.append((next_x, next_y))
                    visited[next_x][next_y] = 1
                    distanceFromX1Y1[(next_x, next_y)] = distanceFromX1Y1[(x, y)] + 1
    return distancesHashtable


def removeCellsInDS(eligibleLeakCells, openCellsInSquare) -> list[(int, int)]:
    for cellInSquare in openCellsInSquare:
        # if cellInSquare in eligibleLeakCells:
        eligibleLeakCells.remove(cellInSquare)  # should we check to make sure cellInSquare is in openCells
    return eligibleLeakCells


kList = []
highestK = 18  # 30 for size 50, 18 for size 30, 15 for size 25, 12 for size 20
for num in range(1, highestK + 1):
    kList.append(num)

winsHashtableBot5 = {}
winsHashtableBot6 = {}
for k in kList:
    winsHashtableBot5[k] = 0
    winsHashtableBot6[k] = 0

iterationsPerK = 250

# Bot 5
for k in kList:
    for count in range(0, iterationsPerK):
        print("K value and count value are as follows: " + str(k) + " " + str(count) + " Bot 5")
        ship5 = Ship()
        arr = ship5.arr
        size = ship5.size
        openCells = getOpenCells(arr, size)

        # pick 2 different locations for bot and leak

        placedLeakOutsideDS = False
        bot = None
        leak1 = None
        leak2 = None
        eligibleBotCells = openCells.copy()
        while not placedLeakOutsideDS:
            # then choose leak out of cells not in bot's detection square
            eligibleLeakCells = openCells.copy()
            indexBot = np.random.randint(0, len(eligibleBotCells))
            bot = eligibleBotCells.pop(indexBot)
            openCellsInSquare = getOpenSquareCells(arr, k, bot, size)
            eligibleLeakCells = removeCellsInDS(eligibleLeakCells, openCellsInSquare)

            # printShip(arr, bot, size)
            if len(eligibleLeakCells) >= 2:  # if there are at least 2 eligible leak cells, place two leaks
                placedLeakOutsideDS = True
                indexLeak1 = np.random.randint(0, len(eligibleLeakCells))
                leak1 = eligibleLeakCells.pop(indexLeak1)
                indexLeak2 = np.random.randint(0, len(eligibleLeakCells))
                leak2 = eligibleLeakCells.pop(indexLeak2)
                arr[leak1[0]][leak1[1]][0] = 3  # mark this cell as the leak
                arr[leak2[0]][leak2[1]][0] = 3  # mark this cell as the leak
            else:
                print("pick another bot location Bot 5")

        print("first leak cell: " + str(leak1) + " second leak cell: " + str(leak2))
        print("initial bot cell: " + str(bot))

        numActions = 0
        numLeaksFound = 0

        while numLeaksFound < 2:
            openCellsAndBool = getOpenCellsAndCheckLeak(arr, k, bot, size)
            openCellsInSquare = openCellsAndBool[0]
            leakDetected = openCellsAndBool[1]
            numActions = numActions + 1
            if leakDetected:
                if arr[bot[0]][bot[1]][0] == 3:  # bot found the leak in its current cell
                    numLeaksFound = numLeaksFound + 1
                    if numLeaksFound == 2:
                        break
                    else:
                        arr[bot[0]][bot[1]][0] = 2  # we found one leak, so ignore this one from now on
                else:
                    arr[bot[0]][bot[1]][0] = 2  # bot didn't find leak, so it must keep searching
                    # we're searching for the last leak, so eliminate all cells outside of square
                    if numLeaksFound == 1:
                        allUnexploredCells = getAllUnexploredOutsideSquare(arr, size, openCellsInSquare)
                        for cell in allUnexploredCells:
                            arr[cell[0]][cell[1]][0] = 2

            else:
                # mark all the unexplored cells in this square as explored and not containing leak
                for cell in openCellsInSquare:
                    arr[cell[0]][cell[1]][0] = 2

            # find next location to move
            closestUnexploredNeighborsOutput = findClosestUnexploredNeighbors(bot[0], bot[1], size, arr)
            closestUnexploredNeighbors = closestUnexploredNeighborsOutput[0]
            indexOpenCell = np.random.randint(0, len(closestUnexploredNeighbors))
            bot = closestUnexploredNeighbors[indexOpenCell]

            # update number of actions: the bot sensed once and moved however many cells
            distMoved = closestUnexploredNeighborsOutput[1]
            numActions = numActions + distMoved

        winsHashtableBot5[k] = winsHashtableBot5[k] + numActions

# Bot 6
for k in kList:
    for count in range(0, iterationsPerK):
        print("K value and count value are as follows: " + str(k) + " " + str(count) + " Bot 6")
        ship6 = Ship()
        arr = ship6.arr
        size = ship6.size
        openCells = getOpenCells(arr, size)

        # pick 2 different locations for bot and leak
        # first pick bot
        placedLeakOutsideDS = False
        bot = None
        leak1 = None
        leak2 = None
        eligibleBotCells = openCells.copy()
        while not placedLeakOutsideDS and leak1 is None and leak2 is None:
            # then choose leak out of cells not in bot's detection square
            eligibleLeakCells = openCells.copy()
            indexBot = np.random.randint(0, len(openCells))
            bot = openCells[indexBot]
            openCellsInSquare = getOpenSquareCells(arr, k, bot, size)
            # if openCellsInSquare:
            for cellInSquare in openCellsInSquare:
                # if cellInSquare in eligibleLeakCells:
                eligibleLeakCells.remove(cellInSquare)  # should we check to make sure cellInSquare is in openCells

            # printShip(arr, bot, size)
            if len(eligibleLeakCells) >= 2:  # if there are at least 2 eligible leak cells, place two leaks
                placedLeakOutsideDS = True
                indexLeak1 = np.random.randint(0, len(eligibleLeakCells))
                leak1 = eligibleLeakCells.pop(indexLeak1)
                indexLeak2 = np.random.randint(0, len(eligibleLeakCells))
                leak2 = eligibleLeakCells.pop(indexLeak2)
                arr[leak1[0]][leak1[1]][0] = 3  # mark this cell as the leak
                arr[leak2[0]][leak2[1]][0] = 3  # mark this cell as the leak
            else:
                print("pick another bot location Bot 6")

        print("leak cell: " + str(leak1) + " second leak " + str(leak2))
        print("initial bot cell: " + str(bot))

        numActions = 0
        numLeaksFound = 0
        pairsMatrix = np.zeros((size, size, size, size))
        pairsMatrix = makePairsMatrix(pairsMatrix, openCells)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numTimesLeakDetected = 0
        while numLeaksFound < 2:
            openCellsAndBool = getOpenCellsAndCheckLeak(arr, k, bot, size)
            openCellsInSquare = openCellsAndBool[0]
            leakDetected = openCellsAndBool[1]
            numActions = numActions + 1
            possibleCellsToMoveToOutput = []
            if leakDetected:
                numTimesLeakDetected = numTimesLeakDetected + 1
                if arr[bot[0]][bot[1]][0] == 3:  # bot found the leak in its current cell
                    numLeaksFound = numLeaksFound + 1
                    if numLeaksFound == 1:  # we've found our first leak, so ignore it from now on
                        arr[bot[0]][bot[1]][0] = 2
                        eliminateOneCell(bot, pairsMatrix, size)  # remove bot from pairs list
                    else:  # we've found our second leak, so bot won and we break
                        break

                else:
                    arr[bot[0]][bot[1]][0] = 2  # bot didn't find leak, so it must keep searching
                    eliminateOneCell(bot, pairsMatrix, size)  # remove from pairs list
                    if numLeaksFound == 1:  # we're searching for the last leak, so eliminate all cells outside of square
                        allUnexploredCells = getAllUnexploredOutsideSquare(arr, size, openCellsInSquare)
                        for cell in allUnexploredCells:
                            arr[cell[0]][cell[1]][0] = 2
                        eliminateCells(pairsMatrix, allUnexploredCells, size)
                possibleCellsToMoveToOutput = findUnexploredNeighborsInSquare(bot[0], bot[1], size, k,
                                                                              numTimesLeakDetected, arr)

            else:
                # mark all the unexplored cells in this square as explored and not containing leak
                eliminateCells(pairsMatrix, openCellsInSquare, size)
                for cell in openCellsInSquare:
                    arr[cell[0]][cell[1]][0] = 2
                possibleCellsToMoveToOutput = findSomeDistUnexploredNeighbors(bot[0], bot[1], size, k,
                                                                              numTimesLeakDetected, arr)

            # find next location to move
            possibleCellsToMoveTo = possibleCellsToMoveToOutput[0]
            indexOpenCell = np.random.randint(0, len(possibleCellsToMoveTo))
            bot = possibleCellsToMoveTo[indexOpenCell]

            # update number of actions: the bot sensed once and moved however many cells
            distMoved = possibleCellsToMoveToOutput[1]
            numActions = numActions + distMoved

        winsHashtableBot6[k] = winsHashtableBot6[k] + numActions

print("bot 5 results:")
for item in winsHashtableBot5:
    index = list(winsHashtableBot5.keys()).index(item)
    print(item, winsHashtableBot5[item] / iterationsPerK)

yVals5 = []
for item in winsHashtableBot5:
    index = list(winsHashtableBot5.keys()).index(item)
    yVals5.append(winsHashtableBot5[item] / iterationsPerK)

print("bot 6 results:")
for item in winsHashtableBot6:
    index = list(winsHashtableBot6.keys()).index(item)
    print(item, winsHashtableBot6[item] / iterationsPerK)

yVals6 = []
for item in winsHashtableBot6:
    index = list(winsHashtableBot6.keys()).index(item)
    yVals6.append(winsHashtableBot6[item] / iterationsPerK)

plt.plot(kList, yVals5, label='Bot 5', color='m')
plt.plot(kList, yVals6, label='Bot 6', color='b')

plt.title("Average number of actions")
plt.xlabel("K values (Integers Between 1 and " + str(highestK) + ")")
plt.ylabel("Average number of actions to arrive to leak")
plt.legend(loc='upper right')
plt.show()
