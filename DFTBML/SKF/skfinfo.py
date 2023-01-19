import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

# interface
class SkfInfo(object):
    
    def __init__(self, name, paramPath):
        elem1, elem2 = name.split('-')
        self.__homo = (elem1 == elem2)
        self.name = name
        with open(os.path.join(paramPath, name + '.skf')) as skfFile:
            # line 1
            line1 = _Parse(next(skfFile))
            gridDist, nGridPoints = line1[0], int(line1[1])
            self.__line1 = {'gridDist': gridDist, 'nGridPoints': nGridPoints}
            self.__distSpace = [gridDist * (i + 1) for i in range(nGridPoints)]
            # line 2
            keys = ['Ed', 'Ep', 'Es', 'SPE', 'Ud', 'Up', 'Us', 'fd', 'fp', 'fs']
            self.__line2 = _LineDict(skfFile, keys) if self.__homo else None
            # line 3
            keys = ['mass', 'c2', 'c3', 'c4', 'c5',
                    'c6', 'c7', 'c8', 'c9', 'rcut',
                    'd1', 'd2', 'd3', 'd4', 'd5',
                    'd6', 'd7', 'd8', 'd9', 'd10']
            self.__line3 = _LineDict(skfFile, keys)
            # line 4 to (4 + nGridPoints - 1)
            keys = ['Hdd0', 'Hdd1', 'Hdd2', 'Hpd0', 'Hpd1',
                    'Hpp0', 'Hpp1', 'Hsd0', 'Hsp0', 'Hss0',
                    'Sdd0', 'Sdd1', 'Sdd2', 'Spd0', 'Spd1',
                    'Spp0', 'Spp1', 'Ssd0', 'Ssp0', 'Sss0']
            self.__skTable = _TableDict(skfFile, nGridPoints, keys)
            # for non-spline repulsive interaction
            coeffNameVec = ['c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
            self.__repCoeffVec = [self.__line3[name] for name in coeffNameVec]
            # spline block
            self.__splineInfo = None
            for line in skfFile:
                if line.strip() == 'Spline':
                    self.__splineInfo = _SplineInfo(skfFile)
                    break
    # end __init__
                    
    def GetSkGrid(self):
        return self.__distSpace
        
    def GetSkData(self, key):
        return self.__distSpace, self.__skTable[key]
        
    def GetSkInt(self, key, dist):
        space, value = self.__distSpace, self.__skTable[key]
        return InterpolatedUnivariateSpline(space, value, k=5, ext=1)(dist)
    
    def GetAtomProp(self, key):
        self.__AssertHomo()
        return self.__line2[key]
    
    def GetRep(self, dist):
        if self.__splineInfo is None:
            rcut = self.__line3['rcut']
            distPow = [(dist - rcut)**power for power in range(2, 10)]
            return np.array(self.__repCoeffVec).dot(distPow)
        else:
            return self.__splineInfo.SplineRep(dist)
    
    def GetRepDeriv1(self, dist):
        if self.__splineInfo is None:
            rcut = self.__line3['rcut']
            distPow = [power * (dist - rcut)**power for power in range(1, 9)]
            return np.array(self.__repCoeffVec).dot(distPow)
        else:
            return self.__splineInfo.SplineRepDeriv1(dist)
    
    def SetAtomProp(self, key, value):
        self.__AssertHomo()
        self.__line2[key] = value
    
    def __AssertHomo(self):
        if not self.__homo:
            raise Exception('Atomic value is in homo file, not %s.' % self.name)


class _SplineInfo(object):
    
    def __init__(self, skfFile):
        # line 2
        line2Val = _Parse(next(skfFile))
        nInt, cutoff = int(line2Val[0]), line2Val[1]
        self.__line2 = {'nInt': nInt, 'cutoff': cutoff}
        # line 3
        keys = ['a1', 'a2', 'a3']
        self.__line3 = _LineDict(skfFile, keys)
        # line 4 to (4 + nInt - 2) and line (4 + nInt - 1)
        self.__splineTable, self.__lastDict = _SplineTable(skfFile, nInt - 1)
        self.__distSpace = [sp['start'] for sp in self.__splineTable]
        self.__distSpace += [self.__lastDict['start']]
    
    def SplineRep(self, dist):
        return self.__SplineRep(dist, _RepClose, _DistPow)
    
    def SplineRepDeriv1(self, dist):
        return self.__SplineRep(dist, _RepCloseDeriv1, _DistPowDeriv1)
    
    def __SplineRep(self, dist, repCloseFunc, distPowFunc):
        if dist < self.__distSpace[0]:
            return repCloseFunc(self.__line3, dist)
        if dist > self.__line2['cutoff']:
            return 0.0
        lastStart = self.__lastDict['start']
        if dist <= lastStart:
            space = self.__distSpace
            index = next(ind for ind, val in enumerate(space) if dist <= val) - 1
            # TODO: Seems to work, but should check: index=-1 if dist = lastStart 
            if index < 0:
                index = 0
            coeff = np.array(self.__splineTable[index]['coeff'])
            distStart = self.__splineTable[index]['start']
            distPow = distPowFunc(distStart, dist, maxPow=3)
        else:
            coeff = np.array(self.__lastDict['coeff'])
            distPow = distPowFunc(lastStart, dist, maxPow=5)
        return coeff.dot(distPow)

def _RepClose(line3, dist):
    return np.exp(-line3['a1'] * dist + line3['a2']) + line3['a3']

def _DistPow(start, dist, maxPow):
    return [(dist - start)**power for power in range(maxPow + 1)]

def _RepCloseDeriv1(line3, dist):
    return -line3['a1'] * np.exp(-line3['a1'] * dist + line3['a2'])

def _DistPowDeriv1(start, dist, maxPow):
    return [0.0] + [(pwr + 1) * (dist - start)**pwr for pwr in range(maxPow)]

def _SplineTable(skfFile, numLines):
    keyListTable = ['start', 'end', 'c0', 'c1', 'c2', 'c3']
    splineTable = []
    for _ in range(numLines):
        lineDict = _LineDict(skfFile, keyListTable)
        coeff = [lineDict[key] for key in keyListTable[2:]]
        splineTable += [{'start': lineDict['start'], 'coeff': coeff}]
    keyListLast = ['start', 'end', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5']
    lastLineDict = _LineDict(skfFile, keyListLast)
    lastCoeff = [lastLineDict[key] for key in keyListLast[2:]]
    lastDict = {'start': lastLineDict['start'], 'coeff': lastCoeff}
    return splineTable, lastDict


def _LineDict(skfFile, keyList):
    return {key: val for key, val in zip(keyList, _Parse(next(skfFile)))}

def _TableDict(skfFile, numLines, keyList):
    table = []
    for _ in range(numLines):
        table += [_Parse(next(skfFile))]
    table = list(map(list, list(zip(*table)))) # transpose
    return {key: val for key, val in zip(keyList, table)}

def _Parse(line):
    splitted = line.replace(',', ' ').replace('\t', ' ').split()
    unrolled = []
    for word in splitted:
        if '*' in word:
            count, string = word.split('*')
            unrolled += [string] * int(count)
        else:
            unrolled += [word]
    return [float(val) for val in unrolled]

