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


'''def printShip(arr, bot, size):
    formatted_arr = np.empty((size, size), dtype='U{:d}'.format(20))
    for row in range(0, size):
        for col in range(0, size):
            if arr[row][col][0] != 0:
                # formattedStr = str(arr[row][col][0]) + "\t" + str(round(arr[row][col][1]))
                formatted_arr[row][col] = f"{round(arr[row][col][1], 3):100}"
            else:
                formatted_arr[row][col] = f"{-1:100}"
    # formatted_values = [f"{value:8}" for value in arr]
    for row in range(0, size):
        for col in range(0, size):
            if bot[0] == row and bot[1] == col:
                print(formatted_arr[row][col], end="")  # it's the bot
            elif arr[row][col][0] == 0:  # it's closed
                print(formatted_arr[row][col], end="")
            elif arr[row][col][0] == 1:  # it's unexplored
                print(formatted_arr[row][col], end="")
            else:
                print(formatted_arr[row][col], end="")  # it's the leak
        print("\n")'''


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
    print("printing current ship: ")
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


def senseBeep(prob) -> bool:
    rand = random.random()
    if rand <= prob:
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
    # arr: ship of size 25x25x2
    # bot: tuple representing (x,y) coordinates
    # possibleLeakCells: list of tuples storing cells that may contain the leak
    tempShip = np.copy(arr) # copy the original ship. make modifications to the copy
    prior = tempShip[bot[0]][bot[1]][1] # probabilities of containing leak are stored in the second index of the third dimension of the array
    tempShip[bot[0]][bot[1]][1] = 0 # the probability of the bot's current cell containing the leak is now 0
    sumWithoutBotCell = 1 - prior # calculate the sum of probabilities of all potential leak cells except the bot's current cell
    if bot in possibleLeakCells:
        possibleLeakCells.remove(bot) # remove the bot from the list of possible leak cells
    for cell in possibleLeakCells:
        cellPrior = tempShip[cell[0]][cell[1]][1] # get the prior of the cell's probability of containing leak
        tempShip[cell[0]][cell[1]][1] = cellPrior / sumWithoutBotCell # update probability
    return tempShip # return the copy of the ship which now has updated probabilities


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
    tempShip = np.copy(arr) # modify a copy of the ship
    denominatorSum = 0
    for cell in possibleLeakCells:
        cellPrior = arr[cell[0]][cell[1]][1]
        distBotToCell = distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))]
        if senseResult: # if beep
            likelihood2 = exp((-1 * alpha) * (distBotToCell - 1))
        else: # if no beep
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
    return tempShip # return modified ship


def getValidOpenVal13(row, col, size, arr) -> list[(int, int)]:
    validNeighbors = getValidNeighbors(row, col, size)
    validOpen = []
    for valid in validNeighbors:
        if arr[valid[0]][valid[1]][0] == 1 or arr[valid[0]][valid[1]][0] == 3:
            validOpen.append(tuple(valid))
    return validOpen


def senseAgainAlongPath(minDistNeighborToHighestProb, distBotToHighestProbCell) -> bool:
    twoThirdsToHighProbCell = distBotToHighestProbCell * (2/3)
    if minDistNeighborToHighestProb <= twoThirdsToHighProbCell:
        return True
    return False


tempList = np.linspace(0, 0.1, 36)
alphaList = np.delete(tempList, 0)

winsHashtableBot3 = {}
winsHashtableBot4 = {}

winsHashtableBot7 = {}
winsHashtableBot8 = {}
winsHashtableBot9 = {}
for alpha in alphaList:
    winsHashtableBot3[alpha] = 0
    winsHashtableBot4[alpha] = 0
    winsHashtableBot7[alpha] = 0
    winsHashtableBot8[alpha] = 0
    winsHashtableBot9[alpha] = 0

iterationsPerAlpha = 100

# Bot 3
for alpha in alphaList:
    for count in range(0, iterationsPerAlpha):
        print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 3")
        ship = Ship()
        arr = ship.arr
        size = ship.size
        openCells = getOpenCells(arr, size)
        possibleLeakCells = openCells.copy()
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then pick leak
        indexLeak = np.random.randint(0, len(openCells))
        leak = openCells[indexLeak]
        arr[leak[0]][leak[1]][0] = 3
        openCells.append(bot)  # put bot back in openCells list
        print("leak location: " + str(leak))

        # third initialize probabilities of all cells containing leak as 1/len(openCells)
        prior_t0 = 1 / len(openCells)
        for cell in openCells:
            arr[cell[0]][cell[1]][1] = prior_t0
        #printShip(arr, bot, size)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numActions = 0

        while bot != leak:
            print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
            # current bot cell doesn't have leak, so update beliefs about other cells
            arr = movement_update(arr, bot, possibleLeakCells)
            #printShip(arr, bot, size) # sense and maybe get a beep
            distBotToLeak = distancesHashtable[((bot[0], bot[1]), (leak[0], leak[1]))] # distance between bot and leak
            likelihood = exp((-1 * alpha) * (distBotToLeak - 1)) # probability of hearing beep
            senseResult = senseBeep(likelihood) # boolean: whether the bot heard a beep
            numActions = numActions + 1
            '''if not senseResult: # if bot didn't hear the beep, change the likelihood
                likelihood = 1 - likelihood'''

            # update probabilities after sensing
            arr = sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha)
            #printShip(arr, bot, size)

            # find cell with max probability of having the leak and move towards it
            maxProb = float('-inf')
            allHighestProbCells = []
            for index in range(len(possibleLeakCells)):
                cell = possibleLeakCells[index]
                if arr[cell[0]][cell[1]][1] == maxProb: # equal probability, so must choose the closest one later
                    maxProb = arr[cell[0]][cell[1]][1]
                    allHighestProbCells.append(cell)
                elif arr[cell[0]][cell[1]][1] > maxProb: # found a higher probability, throw out previous cells
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
                while bot != highestProbCell and bot != leak: # along the path from the bot to the highestProbCell
                    validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                    for neighbor in validOpen:
                        neighborToHighestProb = distancesHashtable[((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                        if neighborToHighestProb < minDistOfNeighborToDest:
                            minDistOfNeighborToDest = neighborToHighestProb
                            nextCell = neighbor
                    # nextCell is the cell that will move the bot closest to the highestProbCell
                    bot = nextCell
                    numActions = numActions + 1
                    # if the bot hasn't arrived to the highestProbCell and isn't at the leak, then do a movement update
                    if bot != highestProbCell and bot != leak:
                        arr = movement_update(arr, bot, possibleLeakCells)
                    #printShip(arr, bot, size)
            else:  # do we need this break condition here? don't think so
                print('uma maça')
                break
            #printShip(arr, bot, size)

        winsHashtableBot3[alpha] = winsHashtableBot3[alpha] + numActions
        print("leak found! num actions = " + str(numActions))

# if you hear a beep, stay in the cell and listen again to update the probabilities again.
# sense along thirds of the paths
# Bot 4
for alpha in alphaList:
    for count in range(0, iterationsPerAlpha):
        print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 4")
        ship = Ship()
        arr = ship.arr
        size = ship.size
        openCells = getOpenCells(arr, size)
        possibleLeakCells = openCells.copy()
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then pick leak
        indexLeak = np.random.randint(0, len(openCells))
        leak = openCells[indexLeak]
        arr[leak[0]][leak[1]][0] = 3
        openCells.append(bot)  # put bot back in openCells list
        print("leak location: " + str(leak))

        # third initialize probabilities of all cells containing leak as 1/len(openCells)
        prior_t0 = 1 / len(openCells)
        for cell in openCells:
            arr[cell[0]][cell[1]][1] = prior_t0
        #printShip(arr, bot, size)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numActions = 0

        while bot != leak:
            print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
            # current bot cell doesn't have leak, so update beliefs about other cells
            arr = movement_update(arr, bot, possibleLeakCells)
            #printShip(arr, bot, size) # sense and maybe get a beep
            distBotToLeak = distancesHashtable[((bot[0], bot[1]), (leak[0], leak[1]))] # distance between bot and leak
            likelihood = exp((-1 * alpha) * (distBotToLeak - 1)) # probability of hearing beep
            senseResult = senseBeep(likelihood) # boolean: whether the bot heard a beep
            numActions = numActions + 1

            # update probabilities after sensing
            arr = sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha)
            #printShip(arr, bot, size)

            # if we hear a beep, sense again to get more info since we may be close
            if senseResult:
                senseResult2 = senseBeep(likelihood)
                arr = sense_update(arr, senseResult2, possibleLeakCells, bot, distancesHashtable, alpha)

            # find cell with max probability of having the leak and move towards it
            maxProb = float('-inf')
            allHighestProbCells = []
            for index in range(len(possibleLeakCells)):
                cell = possibleLeakCells[index]
                if arr[cell[0]][cell[1]][1] == maxProb: # equal probability, so must choose the closest one later
                    maxProb = arr[cell[0]][cell[1]][1]
                    allHighestProbCells.append(cell)
                elif arr[cell[0]][cell[1]][1] > maxProb: # found a higher probability, throw out previous cells
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
                while bot != highestProbCell and bot != leak: # along the path from the bot to the highestProbCell
                    validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                    for neighbor in validOpen:
                        neighborToHighestProb = distancesHashtable[((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                        #smallestDistNeighborToHighestProb = None
                        if neighborToHighestProb < minDistOfNeighborToDest:
                            minDistOfNeighborToDest = neighborToHighestProb
                            #smallestDistNeighborToHighestProb = neighborToHighestProb
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
                    if bot != highestProbCell and bot != leak:
                        arr = movement_update(arr, bot, possibleLeakCells)
                    #printShip(arr, bot, size)
            else:  # do we need this break condition here? don't think so
                print('uma maça')
                break
            #printShip(arr, bot, size)
        winsHashtableBot4[alpha] = winsHashtableBot4[alpha] + numActions
        print("leak found! num actions = " + str(numActions))

'''for alpha in alphaList:
    for count in range(0, iterationsPerAlpha):
        print("Alpha value and count value are as follows: " + str(alpha) + " " + str(count) + " Bot 4")
        ship = Ship()
        arr = ship.arr
        size = ship.size
        openCells = getOpenCells(arr, size)
        possibleLeakCells = openCells.copy()
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then pick leak
        indexLeak = np.random.randint(0, len(openCells))
        leak = openCells[indexLeak]
        arr[leak[0]][leak[1]][0] = 3
        openCells.append(bot)  # put bot back in openCells list
        print("leak location: " + str(leak))
        # third initialize probabilities of all cells containing leak as 1/len(openCells)
        prior_t0 = 1 / len(openCells)
        for cell in openCells:
            arr[cell[0]][cell[1]][1] = prior_t0
        #printShip(arr, bot, size)
        #distancesHashtable = makeDistancesHashtable(arr, size)
        distancesHashtable = BFS_traversal(arr, size, openCells)
        numActions = 0

        while bot != leak:
            print("bot location: " + str(bot) + " with " + str(numActions) + " actions")
            # current bot cell doesn't have leak, so update beliefs about other cells
            arr = movement_update(arr, bot, possibleLeakCells)
            #printShip(arr, bot, size) # sense and maybe get a beep
            # distBotToLeak = findDistanceBetween(bot[0], bot[1], leak[0], leak[1], size, arr)
            distBotToLeak = distancesHashtable[((bot[0], bot[1]), (leak[0], leak[1]))]
            likelihood = exp((-1 * alpha) * (distBotToLeak - 1))
            senseResult = senseBeep(likelihood)
            numActions = numActions + 1
            if not senseResult:
                likelihood = 1 - likelihood

            # update probabilities after sensing
            arr = sense_update(arr, senseResult, possibleLeakCells, bot, distancesHashtable, alpha)

            # find cell with max probability of having the leak and move to it

            maxProb = float('-inf')
            allHighestProbCells = []

            for index in range(len(possibleLeakCells)):
                cell = possibleLeakCells[index]
                if arr[cell[0]][cell[1]][1] == maxProb: # equal probability, so must choose the closest one later
                    maxProb = arr[cell[0]][cell[1]][1]
                    allHighestProbCells.append(cell)
                elif arr[cell[0]][cell[1]][1] > maxProb: # found a higher probability, throw out previous cells
                    maxProb = arr[cell[0]][cell[1]][1]
                    allHighestProbCells = [cell]

            # break ties of highest probability based on distance from bot
            distsHighestProbCells = []
            for cell in allHighestProbCells:
                distsHighestProbCells.append(distancesHashtable[((bot[0], bot[1]), (cell[0], cell[1]))])

            minIndex = -1
            minDistToHighestProb = float('inf')
            nearestHighestProbCell = None
            for count in range(0, len(distsHighestProbCells)):
                if distsHighestProbCells[count] < minDistToHighestProb:
                    minDistToHighestProb = distsHighestProbCells[count]
                    minIndex = count

            # move towards the cell with highest probability of containing leak
            if minIndex != -1:
                highestProbCell = allHighestProbCells[minIndex]  # move to cell with highest probability of containing leak
                nextCell = None
                minDistOfNeighborToDest = float('inf')
                while bot != highestProbCell and bot != leak:
                    validOpen = getValidOpenVal13(bot[0], bot[1], size, arr)
                    smallestDistNeighborToHighestProb = None
                    for neighbor in validOpen:
                        neighborToHighestProb = distancesHashtable[((neighbor[0], neighbor[1]), (highestProbCell[0], highestProbCell[1]))]
                        if neighborToHighestProb < minDistOfNeighborToDest:
                            minDistOfNeighborToDest = neighborToHighestProb
                            smallestDistNeighborToHighestProb = neighborToHighestProb
                            nextCell = neighbor
                    if senseAgainAlongPath(minDistToHighestProb, smallestDistNeighborToHighestProb):
                        bot = nextCell
                        break
                    bot = nextCell
                    numActions = numActions + 1
                    arr = movement_update(arr, bot, possibleLeakCells)
            else:  # do we need this break condition here? don't think so
                break
            #printShip(arr, bot, size)

        winsHashtableBot4[alpha] = winsHashtableBot4[alpha] + numActions
        print("leak found! num actions = " + str(numActions))'''

print("bot 3 results:")
for item in winsHashtableBot3:
    index = list(winsHashtableBot3.keys()).index(item)
    print(item, winsHashtableBot3[item] / iterationsPerAlpha)

yVals3 = []
for item in winsHashtableBot3:
    index = list(winsHashtableBot3.keys()).index(item)
    yVals3.append(winsHashtableBot3[item] / iterationsPerAlpha)

print("bot 4 results:")
for item in winsHashtableBot4:
    index = list(winsHashtableBot4.keys()).index(item)
    print(item, winsHashtableBot4[item] / iterationsPerAlpha)

yVals4 = []
for item in winsHashtableBot4:
    index = list(winsHashtableBot4.keys()).index(item)
    yVals4.append(winsHashtableBot4[item] / iterationsPerAlpha)

plt.plot(alphaList, yVals3, label='Bot 3', color='m')
plt.plot(alphaList, yVals4, label='Bot 4', color='b')

plt.title("Average number of actions")
plt.xlabel("35 Alpha values Between 0.002857 and 0.1")
#plt.xlabel("35 Alpha values Between " + str(round(alphaList[0], 5)) + " and " + str(round(alphaList[-1], 5)))
plt.ylabel("Average number of actions to arrive to leak")
plt.legend(loc='upper right')
plt.show()
