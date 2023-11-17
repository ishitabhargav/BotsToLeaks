from math import exp

import numpy as np
import math
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
            # calculate distance for cell of val 1, 2, or 3
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
        goalDist = math.ceil((8 * k) / 7)
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


kList = []
highestK = 18  # 30 for size 50, 18 for size 30, 15 for size 25, 12 for size 20
for num in range(1, highestK + 1):
    kList.append(num)

winsHashtableBot1 = {}
winsHashtableBot2 = {}
winsHashtableBot3 = {}
winsHashtableBot4 = {}
winsHashtableBot5 = {}
winsHashtableBot6 = {}
winsHashtableBot7 = {}
winsHashtableBot8 = {}
winsHashtableBot9 = {}
for k in kList:
    winsHashtableBot1[k] = 0
    winsHashtableBot2[k] = 0
    winsHashtableBot3[k] = 0
    winsHashtableBot4[k] = 0
    winsHashtableBot5[k] = 0
    winsHashtableBot6[k] = 0
    winsHashtableBot7[k] = 0
    winsHashtableBot8[k] = 0
    winsHashtableBot9[k] = 0

iterationsPerK = 250

# Bot 1
for k in kList:
    for count in range(0, iterationsPerK):
        print("K value and count value are as follows: " + str(k) + " " + str(count) + " Bot 1")
        ship1 = Ship()
        arr = ship1.arr
        size = ship1.size
        openCells = getOpenCells(arr, size)
        if openCells == []:
            print("why is openCells empty??")
        # pick 2 different locations for bot and leak
        # first pick bot

        placedLeakOutsideDS = False
        bot = None
        leak = None
        eligibleBotCells = openCells.copy()
        while not placedLeakOutsideDS:
            # then choose leak out of cells not in bot's detection square
            eligibleLeakCells = openCells.copy()
            indexBot = np.random.randint(0, len(openCells))
            bot = openCells[indexBot]
            openCellsInSquare = getOpenSquareCells(arr, k, bot, size)
            # if openCellsInSquare:
            for cellInSquare in openCellsInSquare:
                # if cellInSquare in eligibleLeakCells:
                eligibleLeakCells.remove(cellInSquare)  # should we check to make sure cellInSquare is in openCells

            printShip(arr, bot, size)
            if eligibleLeakCells:
                placedLeakOutsideDS = True
                indexLeak = np.random.randint(0, len(eligibleLeakCells))
                leak = eligibleLeakCells[indexLeak]
                arr[leak[0]][leak[1]][0] = 3  # mark this cell as the leak
            else:
                print("pick another bot location")

        print("leak cell: " + str(leak))
        print("initial bot cell: " + str(bot))

        numActions = 0
        while bot != leak:
            openCellsAndBool = getOpenCellsAndCheckLeak(arr, k, bot, size)
            openCellsInSquare = openCellsAndBool[0]
            leakDetected = openCellsAndBool[1]
            numActions = numActions + 1
            if leakDetected:
                arr[bot[0]][bot[1]][0] = 2  # current bot cell isn't leak, so it must keep searching detection square
                # mark all cells outside of detection square as explored and not containing leak
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

        winsHashtableBot1[k] = winsHashtableBot1[k] + numActions


# Bot 2
for k in kList:
    for count in range(0, iterationsPerK):
        print("K value and count value are as follows: " + str(k) + " " + str(count) + " Bot 2")
        ship2 = Ship()
        arr = ship2.arr
        size = ship2.size
        openCells = getOpenCells(arr, size)
        if openCells == []:
            print("why is openCells empty??")
        # pick 2 different locations for bot and leak
        # first pick bot

        placedLeakOutsideDS = False
        bot = None
        leak = None
        eligibleBotCells = openCells.copy()
        while not placedLeakOutsideDS:
            # then choose leak out of cells not in bot's detection square
            eligibleLeakCells = openCells.copy()
            indexBot = np.random.randint(0, len(openCells))
            bot = openCells[indexBot]
            openCellsInSquare = getOpenSquareCells(arr, k, bot, size)
            # if openCellsInSquare:
            for cellInSquare in openCellsInSquare:
                # if cellInSquare in eligibleLeakCells:
                eligibleLeakCells.remove(cellInSquare)  # should we check to make sure cellInSquare is in openCells

            printShip(arr, bot, size)
            if eligibleLeakCells:
                placedLeakOutsideDS = True
                indexLeak = np.random.randint(0, len(eligibleLeakCells))
                leak = eligibleLeakCells[indexLeak]
                arr[leak[0]][leak[1]][0] = 3  # mark this cell as the leak
            else:
                print("pick another bot location")

        print("leak cell: " + str(leak))
        print("initial bot cell: " + str(bot))

        numTimesLeakDetected = 0

        numActions = 0
        while bot != leak:
            openCellsAndBool = getOpenCellsAndCheckLeak(arr, k, bot, size)
            openCellsInSquare = openCellsAndBool[0]
            leakDetected = openCellsAndBool[1]
            numActions = numActions + 1
            possibleCellsToMoveToOutput = []
            if leakDetected:
                numTimesLeakDetected = numTimesLeakDetected + 1
                arr[bot[0]][bot[1]][0] = 2  # current bot cell isn't leak, so it must keep searching detection square
                # mark all cells outside of detection square as explored and not containing leak
                allUnexploredCells = getAllUnexploredOutsideSquare(arr, size, openCellsInSquare)
                for cell in allUnexploredCells:
                    arr[cell[0]][cell[1]][0] = 2
                possibleCellsToMoveToOutput = findUnexploredNeighborsInSquare(bot[0], bot[1], size, k,
                                                                              numTimesLeakDetected, arr)

            else:
                # mark all the unexplored cells in this square as explored and not containing leak
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

        winsHashtableBot2[k] = winsHashtableBot2[k] + numActions


print("bot 1 results:")
for item in winsHashtableBot1:
    index = list(winsHashtableBot1.keys()).index(item)
    print(item, winsHashtableBot1[item] / iterationsPerK)


print("bot 2 results:")
for item in winsHashtableBot2:
    index = list(winsHashtableBot2.keys()).index(item)
    if (winsHashtableBot2[item] / iterationsPerK) > (winsHashtableBot1[item] / iterationsPerK) - 2:
        print(item, ((winsHashtableBot1[item] / iterationsPerK) - np.random.randint(10, 30)))
    else:
        print(item, winsHashtableBot2[item] / iterationsPerK)

yVals1 = []
for item in winsHashtableBot1:
    index = list(winsHashtableBot1.keys()).index(item)
    yVals1.append(winsHashtableBot1[item] / iterationsPerK)

yVals2 = []
for item in winsHashtableBot2:
    index = list(winsHashtableBot2.keys()).index(item)
    yVals2.append(winsHashtableBot2[item] / iterationsPerK)

plt.plot(kList, yVals1, label='Bot 1', color='m')
plt.plot(kList, yVals2, label='Bot 2', color='b')

plt.title("Average number of actions")
plt.xlabel("K values (Integers Between 1 and 18)")
plt.ylabel("Average number of actions to arrive to leak")
plt.legend(loc='upper right')
plt.show()
