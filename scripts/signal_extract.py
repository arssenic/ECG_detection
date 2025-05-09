import cv2
import numpy as np
import matplotlib.pyplot as plt
import dataclasses
from dataclasses import dataclass
from math import isnan, sqrt, asin, pi
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, Callable, TypeVar
import numpy as np
import pathlib
from pathlib import Path
import csv

Int = int
Float = float
Numeric = Union[Float, Int]

A = TypeVar("A")
B = TypeVar("B")

@dataclass(frozen=True)
class Point:
    x: Numeric
    y: Numeric

    @property
    def index(self) -> int:
        if isinstance(self.x, int):
            return self.x
        else:
            return round(self.x)

    @property
    def values(self) -> Tuple[Numeric, Numeric]:
        return (self.x, self.y)

@dataclasses.dataclass(frozen=True)
class Image:
    data: np.ndarray

    @property
    def height(self):
        return self.data.shape[0]

    @property
    def width(self):
        return self.data.shape[1]

class ColorImage(Image):
    def toGrayscale(self):
        return GrayscaleImage(cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY))

class GrayscaleImage(Image):
    def toColor(self):
        return ColorImage(cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR))

    def toBinary(self, threshold: Optional[int] = None, inverse: bool = True):
        if threshold is None:
            _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
            binaryData = np.invert(binaryData.astype('bool')).astype('uint8') if inverse else binaryData
        else:
            mode = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
            _, binaryData = cv2.threshold(self.data, threshold, 1, mode)

        return BinaryImage(binaryData)

    def histogram(self):
        counts, _ = np.histogram(self.data, 255, range=(0,255))
        return counts

    def whitePointAdjusted(self, strength: float = 1.0):
        hist = self.histogram()
        whitePoint = np.argmax(hist)
        whiteScaleFactor = 255 / whitePoint * strength
        return GrayscaleImage(cv2.addWeighted(self.data, whiteScaleFactor, self.data, 0, 0))

class BinaryImage(Image):
    def toColor(self):
        return ColorImage(cv2.cvtColor(self.data * 255, cv2.COLOR_GRAY2BGR))

    def toGrayscale(self):
        return GrayscaleImage(self.data * 255)

def findContiguousRegions(oneDimImage: np.ndarray) -> Iterable[Tuple[int, int]]:
    locations = []
    start = None
    for index, pixel in enumerate(oneDimImage):
        if pixel > 0 and start is None:
            start = index
        elif pixel == 0 and start is not None:
            locations.append((start, index))
            start = None

    return locations


def findContiguousRegionCenters(oneDimImage: np.ndarray) -> Iterable[int]:
    return [int(np.mean(list(locationPair))) for locationPair in findContiguousRegions(oneDimImage)]


def euclideanDistance(x: Numeric, y: Numeric) -> float:
    return sqrt((x**2) + (y**2))


def distanceBetweenPoints(firstPoint: Point, secondPoint: Point) -> float:
    return euclideanDistance(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y)


def angleFromOffsets(x: Numeric, y: Numeric) -> float:
    return asin(y/euclideanDistance(x,y)) / pi * 180


def angleBetweenPoints(firstPoint: Point, secondPoint: Point) -> float:
    deltaX = secondPoint.x - firstPoint.x
    deltaY = secondPoint.y - firstPoint.y
    return angleFromOffsets(deltaX, deltaY)


def angleSimilarity(firstAngle: float, secondAngle: float) -> float:
    return (180 - abs(secondAngle - firstAngle)) / 180


def searchArea(initialRow: int, radius: int) -> Iterable[Tuple[int, int]]:
    area = []
    for column in range(1, radius+1):
        verticalOffset = 0
        while euclideanDistance(column, verticalOffset + 1) <= float(radius):
            verticalOffset += 1
        area.append((initialRow - verticalOffset, initialRow + verticalOffset))

    return area

def mapList(elements: Iterable[A], func: Callable[[A], B]) -> 'List[B]':
    return list(map(func, elements))

def getPointLocations(image: np.ndarray) -> List[List[Point]]:
    columns = np.swapaxes(image, 0, 1)

    pointLocations = []

    for column, pixels in enumerate(columns):
        rows = findContiguousRegionCenters(pixels)
        points = mapList(rows, lambda row: Point(column, row))

        pointLocations.append(points)

    return pointLocations


def score(currentPoint: Point, candidatePoint: Point, candidateAngle: float) -> float:
    DISTANCE_WEIGHT = .5

    currentAngle = angleBetweenPoints(candidatePoint, currentPoint)
    angleValue = 1 - angleSimilarity(currentAngle, candidateAngle)
    distanceValue = distanceBetweenPoints(currentPoint, candidatePoint)

    return (distanceValue * DISTANCE_WEIGHT) + (angleValue * (1 - DISTANCE_WEIGHT))

def lowerClamp(value: Numeric, limit: Numeric) -> Numeric:
    return value if (value > limit) else limit

def flatten(listOfLists: Iterable[Iterable[A]]) -> Iterable[A]:
    return (e for _list in listOfLists for e in _list)

def getAdjacent(pointsByColumn, bestPathToPoint, startingColumn: int, minimumLookBack: int):
    rightColumnIndex = startingColumn
    leftColumnIndex = int(lowerClamp(startingColumn-minimumLookBack, 0))

    result = list(flatten(pointsByColumn[leftColumnIndex:rightColumnIndex]))

    while len(result) == 0 and leftColumnIndex >= 0:
        leftColumnIndex -= 1
        result = list(flatten(pointsByColumn[leftColumnIndex:rightColumnIndex])) 

    for point in result:
        assert point in bestPathToPoint, "Found point that hasn't yet been frozen"
        pointScore, _, pointAngle = bestPathToPoint[point]
        yield pointScore, point, pointAngle


def interpolate(fromPoint: Point, toPoint: Point) -> Iterator[Point]:
    slope = (toPoint.y - fromPoint.y) / (toPoint.x - fromPoint.x)
    f = lambda x: slope * (x - toPoint.x) + toPoint.y

    for x in range(fromPoint.index + 1, toPoint.index):
        yield Point(x, f(x))


def convertPointsToSignal(points: List[Point], width: Optional[int] = None) -> np.ndarray:
    assert len(points) > 0

    firstPoint = points[0]  
    lastPoint = points[-1]  

    arraySize = width or (firstPoint.x + 1)
    signal = np.full(arraySize, np.nan, dtype=float)

    signal[firstPoint.index] = firstPoint.y
    priorPoint = firstPoint

    for point in points[1:]:
        if isnan(signal[point.index + 1]):
            for interpolatedPoint in interpolate(point, priorPoint):
                signal[interpolatedPoint.index] = interpolatedPoint.y

        signal[point.index] = point.y
        priorPoint = point

    return signal


def extractSignal(binary: BinaryImage) -> Optional[np.ndarray]:
    pointsByColumn = getPointLocations(binary.data)
    points = list(flatten(pointsByColumn))

    if len(points) == 0:
        return None

    minimumLookBack = 1

    bestPathToPoint: Dict[Point, Tuple[float, Optional[Point], float]] = {}

   
    for column in pointsByColumn[:1]:
        for point in column:
            bestPathToPoint[point] = (0, None, 0)

    for column in pointsByColumn[1:]:
        for point in column:
            adjacent = list(getAdjacent(pointsByColumn, bestPathToPoint, point.index, minimumLookBack))

            if len(adjacent) == 0:
                print(f"None adjacent to {point}")
                bestPathToPoint[point] = (0, None, 0)
            else:
                bestScore: float
                bestPoint: Point
                bestScore, bestPoint = min(
                    [(score(point, candidatePoint, candidateAngle) + cadidateScore, candidatePoint)
                    for cadidateScore, candidatePoint, candidateAngle in adjacent],
                    key=lambda triplet: triplet[0] 
                )
                bestPathToPoint[point] = (bestScore, bestPoint, angleBetweenPoints(bestPoint, point))

    OPTIMAL_ENDING_WIDTH = 20
    optimalCandidates = list(getAdjacent(pointsByColumn, bestPathToPoint, startingColumn=binary.width, minimumLookBack=OPTIMAL_ENDING_WIDTH))

    _, current = min(
        [(totalScore, point) for totalScore, point, _ in optimalCandidates],
        key=lambda pair: pair[0]
    )

    bestPath = []

    while current is not None:
        bestPath.append(current)
        _, current, _ = bestPathToPoint[current]

    signal = convertPointsToSignal(bestPath) 

    scores = [bestPathToPoint[point][0] ** .5 for point in points]
    plt.imshow(binary.toColor().data, cmap='Greys')
    plt.scatter([point.x for point in points], [point.y for point in points], c=scores)
    plt.plot(signal, c='purple')
    plt.show()

    return signal

output_dir = Path("signal_extracted")
signal_data = Path("signal_data")
output_dir.mkdir(exist_ok=True)
signal_data.mkdir(exist_ok=True)

def process_mask_image(image_path):
    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Failed to load {image_path}")
        return
    
    _, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
    binary_image = BinaryImage(binary)

    signal = extractSignal(binary_image)
    if signal is not None:
        x_vals = np.arange(len(signal))  
        y_vals = -signal  
        
        plt.figure(figsize=(10, 4))
        plt.plot(x_vals, y_vals, color='blue')
        plt.title(f"Extracted Signal - {image_path.name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"{image_path.stem}_extracted_signal.png")
        plt.close()
        print(f"Saved: {output_dir / f'{image_path.stem}_signal.png'}")

        csv_file_path = signal_data / f"{image_path.stem}_extracted_signal.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Index', 'Signal Value'])  
            for x, y in zip(x_vals, y_vals):
                writer.writerow([x, y])  

        print(f"Saved signal data to: {csv_file_path}")
    else:
        print(f"No signal found in: {image_path.name}")

input_dir = Path("signal_detected")
for image_file in input_dir.glob("*.png"):
    process_mask_image(image_file)


