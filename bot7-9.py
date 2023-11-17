from math import exp
from typing import Any

import numpy as np
from ship_file import Ship, getValidNeighbors
import matplotlib.pyplot as plt
import random


# FUNCTIONS HERE


def getOpenCells(arr, size) -> list[(int, int)]:
    openCells = []
    for row in range(0, size):
        for col in range(0, size):
            if arr[row][col][0] == 1:
                openCells.append((row, col))
    return openCells


def printShip(arr, bot, size):
    """formatted_arr = np.zeros((size, size, 1))
    for row in range(0, size):
        for col in range(0, size):
            if arr[row][col][0] != 0:
                # formattedStr = str(arr[row][col][0]) + "\t" + str(round(arr[row][col][1]))
                formatted_arr[row][col] = f"{round(arr[row][col][1], 3):100}"
            else:
                formatted_arr[row][col] = f"{-1:100}"
    """
    # formatted_values = [f"{value:8}" for value in arr]
    for row in range(0, size):
        for col in range(0, size):
            '''if bot[0] == row and bot[1] == col:
                print(round(arr[row][col], 3), end="")  # it's the bot'''
            if arr[row][col][0] == 0:  # it's closed
                print(str(-1.000), end="\t")
            elif arr[row][col][0] == 1:  # it's unexplored
                print(round(arr[row][col][1], 3), end="\t")
            else:
                print(round(arr[row][col][1], 3), end="\t")  # it's the leak
        print("\n")


def senseBeep(likelihood1, likelihood2) -> bool:
    rand = random.random()
    if rand <= likelihood1 or rand <= likelihood2:
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


def movement_update(arr, bot, possibleLeakCells) -> np.ndarray[Any, np.dtype[Any]]:
    tempShip = np.copy(arr)
    prior = tempShip[bot[0]][bot[1]][1]
    tempShip[bot[0]][bot[1]][1] = 0
    sumWithoutBotCell = 1 - prior
    if bot in possibleLeakCells:
        possibleLeakCells.remove(bot)
    for cell in possibleLeakCells:
        cellPrior = tempShip[cell[0]][cell[1]][1]
        tempShip[cell[0]][cell[1]][1] = cellPrior / sumWithoutBotCell
    return tempShip


def movement_update_bot8(bot, pairsMatrix, size, arr):
    # pairsMatrix: 4D array of 30x30x30x30 to maintain probabilities for each pair of cells containing the leaks
    # bot: tuple of (x, y) coordinates of bot's location
    # size of the ship = 30
    tempShip = np.copy(pairsMatrix)
    x1 = bot[0]
    y1 = bot[1]
    sum_bot_pairs = 0
    for a in range(size):
        for b in range(size):
            sum_bot_pairs = sum_bot_pairs + tempShip[x1][y1][a][b]
            tempShip[a][b][x1][y1] = 0
            tempShip[x1][y1][a][b] = 0

    sumWithoutBotPairs = 1 - sum_bot_pairs
    omega = 1e-45
    if sumWithoutBotPairs <= omega:
        print("mancana")
    for one in range(size):
        for two in range(size):
            if arr[one][two][0] != 0:
                for three in range(size):
                    for four in range(size):
                        if tempShip[one][two][three][four] >= omega:  # it's a possible pair of leak cells
                            cellPrior = pairsMatrix[one][two][three][four]
                            val = cellPrior / sumWithoutBotPairs
                            if val >= 1:
                                print("naranja")
                            elif val >= omega:
                                tempShip[one][two][three][four] = cellPrior / sumWithoutBotPairs
                                tempShip[three][four][one][two] = cellPrior / sumWithoutBotPairs
                            else:
                                print(val)
                                print(omega)
                                print("cellprior " + str(cellPrior))
                                print("sum " + str(sumWithoutBotPairs))
                                print("pomelo")

    for one in range(size):
        for two in range(size):
            if arr[one][two][0] != 0:
                for three in range(size):
                    for four in range(size):
                        if tempShip[one][two][three][four] < 0:
                            print("move: you're negative bb")
                            break
    return tempShip


def eliminateNonBotPairs(pairsMatrix, size, bot) -> np.ndarray[Any, np.dtype[Any]]:
    # tempShip = np.copy(pairsMatrix)
    # make a blank pairsMatrix
    # loop through pairsMatrix to get pairs with bot cell in them
    # sum pairs of bot cells up
    # update: cellprior/ (1 - sum bot cells)
    x1 = bot[0]
    y1 = bot[1]
    tempShip = np.zeros((size, size, size, size))
    sum_bot_pairs = 0
    for a in range(size):
        for b in range(size):
            tempShip[a][b][x1][y1] = pairsMatrix[a][b][x1][y1]
            tempShip[x1][y1][a][b] = pairsMatrix[a][b][x1][y1]
            sum_bot_pairs = sum_bot_pairs + pairsMatrix[a][b][x1][y1]

    '''sum_non_bot_pairs = 0
    for one in range(size):
        for two in range(size):
            for three in range(size):
                for four in range(size):
                    if (one, two) != bot and (three, four) != bot:
                        sum_non_bot_pairs = sum_non_bot_pairs + pairsMatrix[one][two][three][four]
                        pairsMatrix[one][two][three][four] = 0'''
    for a in range(size):
        for b in range(size):
            cellPrior = pairsMatrix[a][b][x1][y1]
            tempShip[a][b][x1][y1] = cellPrior / sum_bot_pairs
            tempShip[x1][y1][a][b] = cellPrior / sum_bot_pairs
    return tempShip


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


def makeDistancesHashtable(arr, size):
    distancesHashtable = {}
    for x1 in range(size):
        for y1 in range(size):
            for x2 in range(x1, size):
                start_y = y1 if x1 == x2 else 0
                for y2 in range(start_y, size):
                    if arr[x1][y1][0] != 0 and arr[x2][y2][0] != 0:  # make sure both cells are open
                        if x1 == x2 and y1 == y2:
                            distancesHashtable[((x1, y1), (x2, y2))] = 0  # dist to itself is 0
                            distancesHashtable[((x2, y2), (x1, y1))] = 0
                            # print("distance from (" + str(x1) + ", " + str(y1) + ") to (" + str(x2) + ", " + str(y2) +
                            # ") is 0")
                        else:
                            distancesHashtable[((x1, y1), (x2, y2))] = findDistanceBetween(x1, y1, x2, y2, size, arr)
                            distancesHashtable[((x2, y2), (x1, y1))] = distancesHashtable[((x1, y1), (x2, y2))]
                            if distancesHashtable[((x1, y1), (x2, y2))] == -1:
                                distancesHashtable[((x1, y1), (x2, y2))] = float('inf')
                                distancesHashtable[((x2, y2), (x1, y1))] = float('inf')
                            # print("distance from (" + str(x1) + ", " + str(y1) + ") to (" + str(x2) + ", " + str(y2) +
                            # ") is " + str(distancesHashtable[((x1, y1), (x2, y2))]))
    return distancesHashtable


def sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha) -> np.ndarray[Any, np.dtype[Any]]:
    # arr: ship of size 25x25x2
    # senseResult: boolean for whether the bot heard the beep
    # possibleLeakCells: up-to-date list of possible cells containing the leak
    # bot: tuple of x and y coordinates representing the bot's location
    # distancesHashtable stores the distances between each pair of open cells in the ship
    # alpha: current alpha value

    # calculate the sum that will be denominator of Bayes formula, called denominatorSum
    tempShip = np.copy(arr)  # modify a copy of the ship
    denominatorSum = 0
    for cell in possibleLeakCells:
        cellPrior = arr[cell[0]][cell[1]][1]
        distBotToCell = distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))]
        if senseResult:  # if beep
            likelihood2 = exp((-1 * alpha) * (distBotToCell - 1))
        else:  # if no beep
            likelihood2 = 1 - exp((-1 * alpha) * (distBotToCell - 1))
        denominatorSum = denominatorSum + (cellPrior * likelihood2)

    # update probabilities by multiplying priors with likelihood and dividing by denominatorsum
    for cell in possibleLeakCells:
        cellPrior = arr[cell[0]][cell[1]][1]
        distBotToCell = distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))]
        if senseResult:
            likelihood2 = exp((-1 * alpha) * (distBotToCell - 1))
        else:
            likelihood2 = 1 - exp((-1 * alpha) * (distBotToCell - 1))
        tempShip[cell[0]][cell[1]][1] = (cellPrior * likelihood2) / denominatorSum

    return tempShip  # return modified ship


def sense_update_bot8(senseResult, pairsMatrix, bot, distancesHashtable, alpha, size, arr) -> np.ndarray[Any, np.dtype[Any]]:
    # senseResult: boolean for whether the bot heard the beep
    # distancesHashtable stores the distances between each pair of open cells in the ship
    # alpha: current alpha value

    # calculate the sum that will be denominator of Bayes formula, called denominatorSum
    tempShip = np.copy(pairsMatrix)  # modify a copy of the probability matrix
    denominatorSum = 0
    for one in range(size):
        for two in range(size):
            if arr[one][two][0] != 0:
                for three in range(size):
                    for four in range(size):
                        if tempShip[one][two][three][four] > 0:  # it's a possible pair of leak cells
                            prior = tempShip[one][two][three][four]
                            distBotToPossibleLeak1 = distancesHashtable[((bot[0], bot[1]), (one, two))]
                            distBotToPossibleLeak2 = distancesHashtable[((bot[0], bot[1]), (three, four))]
                            hearBeepBcOfLeak1 = exp((-1 * alpha) * (distBotToPossibleLeak1 - 1))
                            hearBeepBcOfLeak2 = exp((-1 * alpha) * (distBotToPossibleLeak2 - 1))
                            prob_beep_bc_two_leaks = hearBeepBcOfLeak1 + hearBeepBcOfLeak2 - (
                                    hearBeepBcOfLeak1 * hearBeepBcOfLeak2)
                            if not senseResult:
                                prob_beep_bc_two_leaks = 1 - prob_beep_bc_two_leaks
                            denominatorSum = denominatorSum + (prob_beep_bc_two_leaks * prior)
    print('')
    for one in range(size):
        for two in range(size):
            if arr[one][two][0] != 0:
                for three in range(size):
                    for four in range(size):
                        if tempShip[one][two][three][four] > 0:
                            prior = pairsMatrix[one][two][three][four]
                            distBotToPossibleLeak1 = distancesHashtable[((bot[0], bot[1]), (one, two))]
                            distBotToPossibleLeak2 = distancesHashtable[((bot[0], bot[1]), (three, four))]
                            hearBeepBcOfLeak1 = exp((-1 * alpha) * (distBotToPossibleLeak1 - 1))
                            hearBeepBcOfLeak2 = exp((-1 * alpha) * (distBotToPossibleLeak2 - 1))
                            prob_beep_bc_two_leaks = hearBeepBcOfLeak1 + hearBeepBcOfLeak2 - (
                                    hearBeepBcOfLeak1 * hearBeepBcOfLeak2)
                            if not senseResult:
                                prob_beep_bc_two_leaks = 1 - prob_beep_bc_two_leaks
                            tempShip[one][two][three][four] = (prob_beep_bc_two_leaks * prior) / denominatorSum
                            tempShip[three][four][one][two] = (prob_beep_bc_two_leaks * prior) / denominatorSum

    for one in range(size):
        for two in range(size):
            if arr[one][two][0] != 0:
                for three in range(size):
                    for four in range(size):
                        if tempShip[one][two][three][four] < 0:
                            #print("sense: you're negative bb")
                            break
        return tempShip


def getValidOpenVal13(row, col, size, arr) -> list[(int, int)]:
    validNeighbors = getValidNeighbors(row, col, size)
    validOpen = []
    for valid in validNeighbors:
        if arr[valid[0]][valid[1]][0] == 1 or arr[valid[0]][valid[1]][0] == 3:
            validOpen.append(tuple(valid))
    return validOpen


def senseAgainAlongPath(minDistToHighestProb, minDistOfNeighborToDest) -> bool:
    halfwayToDest = minDistToHighestProb / 2
    if minDistOfNeighborToDest <= halfwayToDest:
        return True
    return False


def makePairsMatrix(pairsMatrix, openCells) -> np.ndarray[
    Any, np.dtype[Any]]:  # list[[list[(int, int)]]]: # set[Any]: #
    numOpenCellsMinusOne = len(openCells) - 1
    numPairs = (numOpenCellsMinusOne * (numOpenCellsMinusOne + 1)) / 2  # sum of first n natural numbers
    initial_prob_t0 = 1 / numPairs
    for i in range(0, len(openCells)):
        for j in range(i + 1, len(openCells)):
            one = openCells[i]
            two = openCells[j]
            pairsMatrix[one[0]][one[1]][two[0]][two[1]] = initial_prob_t0
            pairsMatrix[two[0]][two[1]][one[0]][one[1]] = initial_prob_t0
    return pairsMatrix


def senseBeepOneLeak(prob) -> bool:
    rand = random.random()
    if rand <= prob:
        return True
    return False


def findSecondLeakBot9(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, secondLeak, size) -> int:
    # 1. make possibleLeak2Cells by traversing pairsMatrix at the bot's indices
    x1 = bot[0]
    y1 = bot[1]
    possibleLeak2Cells = []
    for a in range(size):
        for b in range(size):
            if pairsMatrix[x1][y1][a][b] > 0:
                arr[a][b][1] = pairsMatrix[x1][y1][a][b]
                possibleLeak2Cells.append((a, b))

    # 2. while loop to find the second leak
    print("finding second leak " + str(secondLeak))
    while bot != secondLeak:
        print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
        # current bot cell doesn't have leak, so update beliefs about other cells
        arr = movement_update(arr, bot, possibleLeakCells)
        # printShip(arr, bot, size) # sense and maybe get a beep
        distBotToLeak = distancesHashtable[
            ((bot[0], bot[1]), (secondLeak[0], secondLeak[1]))]  # distance between bot and leak
        likelihood = exp((-1 * alpha) * (distBotToLeak - 1))  # probability of hearing beep
        senseResult = senseBeepOneLeak(likelihood)  # boolean: whether the bot heard a beep
        numActions = numActions + 1

        # update probabilities after sensing
        arr = sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha)
        # printShip(arr, bot, size)

        # if we hear a beep, sense again to get more info since we may be close
        if senseResult:
            senseResult2 = senseBeepOneLeak(likelihood)
            arr = sense_update(arr, senseResult2, possibleLeakCells, bot, distancesHashtable, alpha)

        # find cell with max probability of having the leak and move towards it
        maxProb = float('-inf')
        allHighestProbCells = []
        for index in range(len(possibleLeakCells)):
            cell = possibleLeakCells[index]
            if arr[cell[0]][cell[1]][1] == maxProb:  # equal probability, so must choose the closest one later
                maxProb = arr[cell[0]][cell[1]][1]
                allHighestProbCells.append(cell)
            elif arr[cell[0]][cell[1]][1] > maxProb:  # found a higher probability, throw out previous cells
                maxProb = arr[cell[0]][cell[1]][1]
                allHighestProbCells = [cell]

        # break ties of highest probability based on distance from bot
        distsHighestProbCells = []
        for cell in allHighestProbCells:
            distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])

        minIndex = -1
        minDist = float('inf')
        nearestHighestProbCell = None
        highProbCellsClosestToBot = []
        for counter in range(0, len(distsHighestProbCells)):
            if distsHighestProbCells[counter] <= minDist:
                minDist = distsHighestProbCells[counter]
                highProbCellsClosestToBot.append(allHighestProbCells[counter])
                minIndex = counter

        # move towards the cell with highest probability of containing leak
        if highProbCellsClosestToBot:
            # randomly choose a cell that's at the closest dist to bot with highest probability of containing leak
            highestProbCell = allHighestProbCells[np.random.randint(0, len(allHighestProbCells))]
            distBotToHighestProbCell = distancesHashtable[((highestProbCell[0], highestProbCell[1]), (bot[0], bot[1]))]
            nextCell = None
            # then pick which of the bot's neighbors is closest to the highestProbCell it's moving towards
            minDistOfNeighborToDest = float('inf')
            while bot != highestProbCell and bot != secondLeak:  # along the path from the bot to the highestProbCell
                validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                for neighbor in validOpen:
                    neighborToHighestProb = distancesHashtable[
                        ((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                    # smallestDistNeighborToHighestProb = None
                    if neighborToHighestProb < minDistOfNeighborToDest:
                        minDistOfNeighborToDest = neighborToHighestProb
                        # smallestDistNeighborToHighestProb = neighborToHighestProb
                        nextCell = neighbor
                bot = nextCell
                numActions = numActions + 1
                # nextCell is the cell that will move the bot closest to the highestProbCell
                # if we are a third of the way to the highest prob cell, break this loop to sense again and determine
                # the highest probability cell again based on beep info
                if senseAgainAlongPath(minDistOfNeighborToDest, distBotToHighestProbCell):
                    bot = nextCell
                    break

                # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                if bot != highestProbCell and bot != secondLeak:
                    arr = movement_update(arr, bot, possibleLeakCells)
    return numActions


def findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, secondLeak, size) -> int:
    # 1. make possibleLeak2Cells by traversing pairsMatrix at the bot's indices
    x1 = bot[0]
    y1 = bot[1]
    possibleLeak2Cells = []
    for a in range(size):
        for b in range(size):
            if pairsMatrix[x1][y1][a][b] > 0:
                arr[a][b][1] = pairsMatrix[x1][y1][a][b]
                possibleLeak2Cells.append((a, b))

    # 2. while loop to find the second leak
    print("finding second leak " + str(secondLeak))
    while bot != secondLeak:
        print("bot3 location: " + str(bot) + " with " + str(numActions) + " actions")
        # current bot cell doesn't have leak, so update beliefs about other cells
        arr = movement_update(arr, bot, possibleLeak2Cells)
        # printShip(arr, bot, size) # sense and maybe get a beep
        distBotToLeak = distancesHashtable[
            ((bot[0], bot[1]), (secondLeak[0], secondLeak[1]))]  # distance between bot and leak
        likelihood = exp((-1 * alpha) * (distBotToLeak - 1))  # probability of hearing beep
        senseResult = senseBeepOneLeak(likelihood)  # boolean: whether the bot heard a beep
        numActions = numActions + 1

        # update probabilities after sensing
        arr = sense_update(arr, senseResult, possibleLeak2Cells, bot, distancesHashtable, alpha)
        # printShip(arr, bot, size)

        # find cell with max probability of having the leak and move towards it
        maxProb = float('-inf')
        allHighestProbCells = []
        for index in range(len(possibleLeak2Cells)):
            cell = possibleLeak2Cells[index]
            if arr[cell[0]][cell[1]][1] == maxProb:  # equal probability, so must choose the closest one later
                maxProb = arr[cell[0]][cell[1]][1]
                allHighestProbCells.append(cell)
            elif arr[cell[0]][cell[1]][1] > maxProb:  # found a higher probability, throw out previous cells
                maxProb = arr[cell[0]][cell[1]][1]
                allHighestProbCells = [cell]

        # break ties of highest probability based on distance from bot
        distsHighestProbCells = []
        for cell in allHighestProbCells:
            distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])

        minIndex = -1
        minDist = float('inf')
        nearestHighestProbCell = None
        highProbCellsClosestToBot = []
        for counter in range(0, len(distsHighestProbCells)):
            if distsHighestProbCells[counter] <= minDist:
                minDist = distsHighestProbCells[counter]
                highProbCellsClosestToBot.append(allHighestProbCells[counter])
                minIndex = counter

        # move towards the cell with highest probability of containing leak
        if highProbCellsClosestToBot:
            # randomly choose a cell that's at the closest dist to bot with highest probability of containing leak
            highestProbCell = allHighestProbCells[np.random.randint(0, len(allHighestProbCells))]
            nextCell = None
            # then pick which of the bot's neighbors is closest to the highestProbCell it's moving towards
            minDistOfNeighborToDest = float('inf')
            while bot != highestProbCell and bot != secondLeak:  # along the path from the bot to the highestProbCell
                validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                for neighbor in validOpen:
                    neighborToHighestProb = distancesHashtable[
                        ((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                    if neighborToHighestProb < minDistOfNeighborToDest:
                        minDistOfNeighborToDest = neighborToHighestProb
                        nextCell = neighbor
                # nextCell is the cell that will move the bot closest to the highestProbCell
                bot = nextCell
                # print("2nd leak bot location: " + str(bot) + " action: " + str(numActions))
                numActions = numActions + 1
                # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                if bot != highestProbCell and bot != secondLeak:
                    arr = movement_update(arr, bot, possibleLeak2Cells)
                # printShip(arr, bot, size)

    return numActions


tempList = np.linspace(0, 0.1, 16)
alphaList = np.delete(tempList, 0)

winsHashtableBot7 = {}
winsHashtableBot8 = {}
winsHashtableBot9 = {}
for alpha in alphaList:
    winsHashtableBot7[alpha] = 0
    winsHashtableBot8[alpha] = 0
    winsHashtableBot9[alpha] = 0

iterationsPerAlpha = 20

# Bot 7
for alpha in alphaList:
   for count in range(0, iterationsPerAlpha):
       print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 7")
       ship = Ship()
       arr = ship.arr
       size = ship.size
       openCells = getOpenCells(arr, size)
       possibleLeakCells = openCells.copy()
       # first pick bot
       indexBot = np.random.randint(0, len(openCells))
       bot = openCells.pop(indexBot)


       # then pick 2 leaks
       indexLeak = np.random.randint(0, len(openCells))
       leak1 = openCells.pop(indexLeak)
       arr[leak1[0]][leak1[1]][0] = 3
       indexLeak2 = np.random.randint(0, len(openCells))
       leak2 = openCells.pop(indexLeak2)
       arr[leak2[0]][leak2[1]][0] = 3


       # put bot and leaks back in openCells list
       openCells.append(bot)
       openCells.append(leak1)
       openCells.append(leak2)
       print("leak location: " + str(leak1) + " second leak " + str(leak2))
       # third initialize probabilities of all cells containing leak as 1/len(openCells)
       prior_t0 = 1 / len(openCells)
       for cell in openCells:
           arr[cell[0]][cell[1]][1] = prior_t0
       # printShip(arr, bot, size)
       distancesHashtable = BFS_traversal(arr, size, openCells)
       numActions = 0
       numLeaksFound = 0
       leak1Found = False
       leak2Found = False
       while not leak1Found or not leak2Found: #numLeaksFound < 2:
           print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
           if bot == leak1:
               leak1Found = True
           elif bot == leak2:
               leak2Found = True
           if leak1Found and leak2Found:
               break
           # current bot cell doesn't have leak, so update beliefs about other cells
           arr = movement_update(arr, bot, possibleLeakCells)
           # printShip(arr, bot, size) # sense and maybe get a beep
           distBotToLeak1 = distancesHashtable[((bot[0], bot[1]), (leak1[0], leak1[1]))]
           distBotToLeak2 = distancesHashtable[((bot[0], bot[1]), (leak2[0], leak2[1]))]
           likelihood1 = exp((-1 * alpha) * (distBotToLeak1 - 1))
           likelihood2 = exp((-1 * alpha) * (distBotToLeak2 - 1))
           senseResult = senseBeep(likelihood1, likelihood2)
           numActions = numActions + 1
           # update probabilities after sensing
           arr = sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha)

           # find cell with max probability of having the leak and move towards it
           maxProb = float('-inf')
           allHighestProbCells = []
           for index in range(len(possibleLeakCells)):
               cell = possibleLeakCells[index]
               if arr[cell[0]][cell[1]][1] == maxProb:  # equal probability, so must choose the closest one later
                   maxProb = arr[cell[0]][cell[1]][1]
                   allHighestProbCells.append(cell)
               elif arr[cell[0]][cell[1]][1] > maxProb:  # found a higher probability, throw out previous cells
                   maxProb = arr[cell[0]][cell[1]][1]
                   allHighestProbCells = [cell]


           # break ties of highest probability based on distance from bot
           distsHighestProbCells = []
           for cell in allHighestProbCells:
               distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])


           minIndex = -1
           minDist = float('inf')
           nearestHighestProbCell = None
           highProbCellsClosestToBot = []
           for counter in range(0, len(distsHighestProbCells)):
               if distsHighestProbCells[counter] <= minDist:
                   minDist = distsHighestProbCells[counter]
                   highProbCellsClosestToBot.append(allHighestProbCells[counter])
                   minIndex = counter


           # move towards the cell with highest probability of containing leak
           if highProbCellsClosestToBot:
               # randomly choose a cell that's at the closest dist to bot with highest probability of containing leak
               highestProbCell = allHighestProbCells[np.random.randint(0, len(allHighestProbCells))]
               nextCell = None
               # then pick which of the bot's neighbors is closest to the highestProbCell it's moving towards
               minDistOfNeighborToDest = float('inf')
               while bot != highestProbCell:  # along the path from the bot to the highestProbCell
                   validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                   for neighbor in validOpen:
                       neighborToHighestProb = distancesHashtable[
                           ((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                       if neighborToHighestProb < minDistOfNeighborToDest:
                           minDistOfNeighborToDest = neighborToHighestProb
                           nextCell = neighbor
                   # nextCell is the cell that will move the bot closest to the highestProbCell
                   bot = nextCell
                   numActions = numActions + 1
                   # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                   if bot != highestProbCell and bot != leak1 and bot != leak2:
                       arr = movement_update(arr, bot, possibleLeakCells)


                   # printShip(arr, bot, size)
           # printShip(arr, bot, size)


       winsHashtableBot7[alpha] = winsHashtableBot7[alpha] + numActions
       print("leak found! num actions = " + str(numActions))

# Bot 8
for alpha in alphaList:
    for count in range(0, iterationsPerAlpha):
        print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 8")
        ship = Ship()
        arr = ship.arr
        size = ship.size
        openCells = getOpenCells(arr, size)
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then pick 2 leaks
        indexLeak = np.random.randint(0, len(openCells))
        leak1 = openCells.pop(indexLeak)
        arr[leak1[0]][leak1[1]][0] = 3
        indexLeak2 = np.random.randint(0, len(openCells))
        leak2 = openCells.pop(indexLeak2)
        arr[leak2[0]][leak2[1]][0] = 3

        # put bot and leaks back in openCells list
        openCells.append(bot)
        openCells.append(leak1)
        openCells.append(leak2)
        print("leak location: " + str(leak1) + " second leak " + str(leak2))
        # third initialize probabilities of all cells containing leak as 1/(num pairs of possible leak cells)
        pairsMatrix = np.zeros((size, size, size, size))
        pairsMatrix = makePairsMatrix(pairsMatrix, openCells)
        # printShip(arr, bot, size)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numActions = 0
        numLeaksFound = 0
        leak1Found = False
        leak2Found = False
        while not leak1Found or not leak2Found:
            print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
            # current bot cell doesn't have leak, so update beliefs about other cells
            if bot == leak1:
                leak1Found = True
                pairsMatrix = eliminateNonBotPairs(pairsMatrix, size, bot)
            elif bot == leak2:
                leak2Found = True
                pairsMatrix = eliminateNonBotPairs(pairsMatrix, size, bot)
            else:
                pairsMatrix = movement_update_bot8(bot, pairsMatrix, size, arr)
            if leak1Found and leak2Found:
                break
            if leak1Found and not leak2Found:
                # find leak2 as in bot 3
                numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, leak2,
                                                size)
                break
            elif leak2Found and not leak1Found:
                # find leak1 as in bot 3
                numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, leak1,
                                                size)
                break
            # printShip(arr, bot, size) # sense and maybe get a beep
            distBotToLeak1 = distancesHashtable[((bot[0], bot[1]), (leak1[0], leak1[1]))]
            distBotToLeak2 = distancesHashtable[((bot[0], bot[1]), (leak2[0], leak2[1]))]
            likelihood1 = exp((-1 * alpha) * (distBotToLeak1 - 1))
            likelihood2 = exp((-1 * alpha) * (distBotToLeak2 - 1))
            senseResult = senseBeep(likelihood1, likelihood2)
            numActions = numActions + 1
            # update probabilities after sensing
            pairsMatrix = sense_update_bot8(senseResult, pairsMatrix, bot, distancesHashtable, alpha, size, arr)
            prevCell = bot

            # find cell with max probability of having the leak
            allHighestProbCells = []
            maxProb = float('-inf')
            for one in range(size):
                for two in range(size):
                    if arr[one][two][0] != 0:
                        for three in range(size):
                            for four in range(size):
                                if pairsMatrix[one][two][three][four] == maxProb and pairsMatrix[one][two][three][four] > 0:
                                    if (one, two) not in allHighestProbCells:
                                        allHighestProbCells.append((one, two))
                                    if (three, four) not in allHighestProbCells:
                                        allHighestProbCells.append((three, four))
                                    # maxPairs.append([(one, two), (three, four)])
                                elif pairsMatrix[one][two][three][four] >= maxProb and pairsMatrix[one][two][three][four] > 0:
                                    maxProb = pairsMatrix[one][two][three][four]
                                    allHighestProbCells = [(one, two), (three, four)]
                                    # maxPairs = [[(one, two), (three, four)]]

            # break ties of highest probability based on dist from bot
            distsHighestProbCells = []
            for cell in allHighestProbCells:
                distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])

            minIndex = -1
            minDist = float('inf')
            highProbCellsClosestToBot = []
            for counter in range(0, len(distsHighestProbCells)):
                if distsHighestProbCells[counter] <= minDist:
                    minDist = distsHighestProbCells[counter]
                    highProbCellsClosestToBot.append(allHighestProbCells[counter])
                    minIndex = counter

            if highProbCellsClosestToBot:
                # randomly choose a cell that's at the closest dist to bot with highest probability of containing leak
                highestProbCell = allHighestProbCells[np.random.randint(0, len(allHighestProbCells))]
                print("highestProbCell: " + str(highestProbCell))
                nextCell = None
                # then pick which of the bot's neighbors is closest to the highestProbCell it's moving towards
                minDistOfNeighborToDest = float('inf')
                while bot != highestProbCell:  # along the path from the bot to the highestProbCell
                    validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                    for neighbor in validOpen:
                        neighborToHighestProb = distancesHashtable[
                            ((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                        if neighborToHighestProb < minDistOfNeighborToDest:
                            minDistOfNeighborToDest = neighborToHighestProb
                            nextCell = neighbor
                    # nextCell is the cell that will move the bot closest to the highestProbCell
                    bot = nextCell
                    print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
                    numActions = numActions + 1
                    # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                    if bot != highestProbCell and bot != leak1 and bot != leak2:
                        pairsMatrix = movement_update_bot8(bot, pairsMatrix, size, arr)
                    elif bot == leak1 and not leak2Found:
                        leak1Found = True
                        numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable,
                                                        leak2, size)
                        leak2Found = True
                        break
                    elif bot == leak2 and not leak1Found:
                        leak2Found = True
                        numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable,
                                                        leak1, size)
                        leak1Found = True
                        break
                    elif leak2Found and leak1Found:
                        break
                    #printShip(arr, bot, size)
            if bot == prevCell:
                print('pineapple')

        winsHashtableBot8[alpha] = winsHashtableBot8[alpha] + numActions
        print("leak found! num actions = " + str(numActions))

# Bot 9
for alpha in alphaList:
    for count in range(0, iterationsPerAlpha):
        print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 9")
        ship = Ship()
        arr = ship.arr
        size = ship.size
        openCells = getOpenCells(arr, size)
        possibleLeakCells = openCells.copy()
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then pick 2 leaks
        indexLeak = np.random.randint(0, len(openCells))
        leak1 = openCells.pop(indexLeak)
        arr[leak1[0]][leak1[1]][0] = 3
        indexLeak2 = np.random.randint(0, len(openCells))
        leak2 = openCells.pop(indexLeak2)
        arr[leak2[0]][leak2[1]][0] = 3

        # put bot and leaks back in openCells list
        openCells.append(bot)
        openCells.append(leak1)
        openCells.append(leak2)
        print("leak location: " + str(leak1) + " second leak " + str(leak2))
        # third initialize probabilities of all cells containing leak as 1/(num pairs of possible leak cells)
        pairsMatrix = np.zeros((size, size, size, size))
        pairsMatrix = makePairsMatrix(pairsMatrix, openCells)
        # printShip(arr, bot, size)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numActions = 0
        numLeaksFound = 0
        leak1Found = False
        leak2Found = False
        while not leak1Found or not leak2Found:
            print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
            # current bot cell doesn't have leak, so update beliefs about other cells
            if bot == leak1:
                leak1Found = True
                pairsMatrix = eliminateNonBotPairs(pairsMatrix, size, bot)
            elif bot == leak2:
                leak2Found = True
                pairsMatrix = eliminateNonBotPairs(pairsMatrix, size, bot)
            else:
                pairsMatrix = movement_update_bot8(bot, pairsMatrix, size, arr)
            if leak1Found and leak2Found:
                break
            if leak1Found and not leak2Found:
                # find leak2 as in bot 3
                numActions = findSecondLeakBot9(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, leak2,
                                                size)
                break
            elif leak2Found and not leak1Found:
                # find leak1 as in bot 3
                numActions = findSecondLeakBot9(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable, leak1,
                                                size)
                break
            # printShip(arr, bot, size) # sense and maybe get a beep
            distBotToLeak1 = distancesHashtable[((bot[0], bot[1]), (leak1[0], leak1[1]))]
            distBotToLeak2 = distancesHashtable[((bot[0], bot[1]), (leak2[0], leak2[1]))]
            likelihood1 = exp((-1 * alpha) * (distBotToLeak1 - 1))
            likelihood2 = exp((-1 * alpha) * (distBotToLeak2 - 1))
            senseResult = senseBeep(likelihood1, likelihood2)
            numActions = numActions + 1
            # update probabilities after sensing
            pairsMatrix = sense_update_bot8(senseResult, pairsMatrix, bot, distancesHashtable, alpha, size, arr)

            # if we hear a beep, sense again to get more info since we may be close

            # find cell with max probability of having the leak
            allHighestProbCells = []
            maxProb = float('-inf')
            for one in range(size):
                for two in range(size):
                    if arr[one][two][0] != 0:
                        for three in range(size):
                            for four in range(size):
                                if pairsMatrix[one][two][three][four] == maxProb and pairsMatrix[one][two][three][four] > 0:
                                    if (one, two) not in allHighestProbCells:
                                        allHighestProbCells.append((one, two))
                                    if (three, four) not in allHighestProbCells:
                                        allHighestProbCells.append((three, four))
                                    # maxPairs.append([(one, two), (three, four)])
                                elif pairsMatrix[one][two][three][four] >= maxProb and pairsMatrix[one][two][three][
                                    four] > 0:
                                    maxProb = pairsMatrix[one][two][three][four]
                                    allHighestProbCells = [(one, two), (three, four)]

            # break ties of highest probability based on dist from bot
            distsHighestProbCells = []
            for cell in allHighestProbCells:
                distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])

            minIndex = -1
            minDist = float('inf')
            highProbCellsClosestToBot = []
            for counter in range(0, len(distsHighestProbCells)):
                if distsHighestProbCells[counter] <= minDist:
                    minDist = distsHighestProbCells[counter]
                    highProbCellsClosestToBot.append(allHighestProbCells[counter])
                    minIndex = counter

            if highProbCellsClosestToBot:
                # randomly choose a cell that's at the closest dist to bot with highest probability of containing leak
                highestProbCell = allHighestProbCells[np.random.randint(0, len(allHighestProbCells))]
                distBotToHighestProbCell = distancesHashtable[
                    ((highestProbCell[0], highestProbCell[1]), (bot[0], bot[1]))]
                nextCell = None
                # then pick which of the bot's neighbors is closest to the highestProbCell it's moving towards
                minDistOfNeighborToDest = float('inf')
                while bot != highestProbCell:  # along the path from the bot to the highestProbCell
                    validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                    for neighbor in validOpen:
                        neighborToHighestProb = distancesHashtable[
                            ((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                        if neighborToHighestProb < minDistOfNeighborToDest:
                            minDistOfNeighborToDest = neighborToHighestProb
                            nextCell = neighbor
                    # nextCell is the cell that will move the bot closest to the highestProbCell
                    bot = nextCell
                    numActions = numActions + 1
                    if senseAgainAlongPath(minDistOfNeighborToDest, distBotToHighestProbCell):
                        bot = nextCell
                        break
                    # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                    if bot != highestProbCell and bot != leak1 and bot != leak2:
                        pairsMatrix = movement_update_bot8(bot, pairsMatrix, size, arr)
                    elif bot == leak1 and not leak2Found:
                        leak1Found = True
                        numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable,
                                                        leak2, size)
                        leak2Found = True
                        break
                    elif bot == leak2 and not leak1Found:
                        leak2Found = True
                        numActions = findSecondLeakBot8(pairsMatrix, arr, bot, numActions, alpha, distancesHashtable,
                                                        leak1, size)
                        leak1Found = True
                        break
                    elif leak2Found and leak1Found:
                        break
                    # printShip(arr, bot, size)

        winsHashtableBot9[alpha] = winsHashtableBot9[alpha] + numActions
        print("leak found! num actions = " + str(numActions))

print("bot 7 results:")
for item in winsHashtableBot7:
    index = list(winsHashtableBot7.keys()).index(item)
    print(item, winsHashtableBot7[item] / iterationsPerAlpha)

yVals7 = []
for item in winsHashtableBot7:
    index = list(winsHashtableBot7.keys()).index(item)
    yVals7.append(winsHashtableBot7[item] / iterationsPerAlpha)

print("bot 8 results:")
for item in winsHashtableBot8:
    index = list(winsHashtableBot8.keys()).index(item)
    print(item, winsHashtableBot8[item] / iterationsPerAlpha)

yVals8 = []
for item in winsHashtableBot8:
    index = list(winsHashtableBot8.keys()).index(item)
    yVals8.append(winsHashtableBot8[item] / iterationsPerAlpha)

print("bot 9 results:")
for item in winsHashtableBot9:
    index = list(winsHashtableBot9.keys()).index(item)
    print(item, winsHashtableBot9[item] / iterationsPerAlpha)

yVals9 = []
for item in winsHashtableBot9:
    index = list(winsHashtableBot9.keys()).index(item)
    yVals9.append(winsHashtableBot9[item] / iterationsPerAlpha)

plt.plot(alphaList, yVals7, label='Bot 7', color='m')
plt.plot(alphaList, yVals8, label='Bot 8', color='b')
plt.plot(alphaList, yVals9, label='Bot 9', color='g')

plt.title("Average number of actions")
plt.xlabel("15 Alpha values Between 0.00667 and 0.1")
# plt.xlabel("35 Alpha values Between 0.002857 and 0.1")
plt.ylabel("Average number of actions to arrive to leak")
plt.legend(loc='upper right')
plt.show()
