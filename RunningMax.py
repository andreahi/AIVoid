import random

def test():
    runningMax = 0.0
    runningAverage = 0.0
    for i in range(0, 10000):
        randint = random.randint(0, 1000)

        runningMax = nextRunningMax(randint, runningMax)
        runningAverage = runningAverage * 0.9 + randint * 0.1
        print "randint: ", randint
        print "runningMax: ", runningMax
        print "runningAvg: ", runningAverage
        print


def nextRunningMax(value, runningMax):
    runningMax = runningMax + 0.1 * (max(runningMax, value) * 0.99 + min(runningMax, value) * 0.01 - runningMax)
    return runningMax

