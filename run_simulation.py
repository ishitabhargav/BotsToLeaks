import numpy as np
from ship_file import Ship, getValidNeighbors
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

    for row in range(topX, bottomX):
        for col in range(leftY, rightY):
            if arr[row][col][0] == 1: # open, unexplored cell
                openCellsInSquare.append((row,col))
    return openCellsInSquare


def getOpenCellsAndCheckLeak(arr, k, bot, size) -> list[list[(int, int)], bool]:
    openCellsInSquare = []
    leakDetected = False
    botX, botY = bot
    topX = getInBoundsVal(botX - k, size)
    bottomX = getInBoundsVal(botX + k, size)
    rightY = getInBoundsVal(botY + k, size)
    leftY = getInBoundsVal(botY - k, size)

    for row in range(topX, bottomX):
        for col in range(leftY, rightY):
            if arr[row][col][0] == 1: # open, unexplored cell
                openCellsInSquare.append((row,col))
            elif arr[row][col][0] == 3: # leak cell!
                openCellsInSquare.append((row, col))
                leakDetected = True
    return [openCellsInSquare, leakDetected]


def findClosestUnexploredNeighbors(botRow, botCol, size, arr) -> list[(int, int)]:
    unexploredNeighbors = []
    queue = [(botRow, botCol)]
    distanceFromBot = {(botRow, botCol): 0}
    visited = np.zeros((size, size))
    visited[botRow][botCol] = 1
    currDistance = 0
    distOfClosestUnexplored = 0 # represents distance of neighbors that will be returned...maybe rename variable
    noNeighborsYet = True
    while not unexploredNeighbors or (unexploredNeighbors and currDistance == distOfClosestUnexplored):
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
    return unexploredNeighbors


kList = []

for num in range(1, 24):
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

# Bot 1
iterationsPerK = 100
for k in kList:
    for count in range(1, iterationsPerK):
        ship1 = Ship()
        arr = ship1.arr
        size = 50
        openCells = ship1.openCells
        # pick 2 different locations for bot and leak
        # first pick bot
        indexBot = np.random.randint(0, len(openCells))
        bot = openCells.pop(indexBot)

        # then choose leak out of cells not in bot's detection square
        openCellsInSquare = [getOpenSquareCells(arr, k, bot, size)]
        eligibleLeakCells = openCells
        if openCellsInSquare:
            for cellInSquare in openCellsInSquare:
                eligibleLeakCells.remove(cellInSquare)  # should we check to make sure cellInSquare is in openCells

        indexLeak = np.random.randint(0, len(eligibleLeakCells))
        leak = eligibleLeakCells[indexLeak]
        arr[leak[0]][leak[1]][0] = 3  # mark this cell as the leak
        openCells.append(bot) # add the bot cell back into the openCells list
        firstTimeStep = True

        while bot != leak:
            # at the first time step, we know the leak isn't in the detection square, so don't check
            if not firstTimeStep: # check if leak is in detection square, increment actions taken by 1
                openCellsAndBool = getOpenCellsAndCheckLeak(arr, k, bot, size)
                openCellsInSquare = openCellsAndBool[0]
                leakDetected = openCellsAndBool[1]
                if leakDetected:
                    print("h")
                    # run some code
                else:
                    print("h")

            firstTimeStep = False
