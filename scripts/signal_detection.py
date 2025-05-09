import cv2
import numpy as np
import dataclasses
from pathlib import Path
from typing import Callable,Union, List, Tuple, Optional,Iterator
import scipy.signal
import scipy.interpolate

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

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 3 and self.data.shape[2] == 3

    def toGrayscale(self):  
        return GrayscaleImage(
            cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
        )


class GrayscaleImage(Image):

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 2

    def toColor(self) -> ColorImage:
        return ColorImage(cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR))

    def toBinary(self, threshold: Optional[int] = None, inverse: bool =True):  
        if threshold is None:
            if inverse:
                binaryData: np.ndarray
                _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
                binaryData = np.invert(binaryData.astype('bool')).astype('uint8')
            else:
                _, binaryData = cv2.threshold(self.data, 0, 1, cv2.THRESH_OTSU)
        else:
            if inverse:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY_INV)
            else:
                _, binaryData = cv2.threshold(self.data, threshold, 1, cv2.THRESH_BINARY)

        return BinaryImage(binaryData)

    def normalized(self): 
        assert self.data.dtype is np.dtype('uint8')
        return GrayscaleImage(self.data / 255)

    def whitePointAdjusted(self, strength: float = 1.0):  
        hist = self.histogram()
        whitePoint = np.argmax(hist)
        whiteScaleFactor = 255 / whitePoint * strength
        return GrayscaleImage(cv2.addWeighted(self.data, whiteScaleFactor, self.data, 0, 0))

    def histogram(self) -> np.ndarray:
        counts, _ = np.histogram(self.data, 255, range=(0,255))
        return counts


class BinaryImage(Image):

    def __post_init__(self) -> None:
        assert isinstance(self.data, np.ndarray)
        assert len(self.data.shape) == 2

    def toColor(self) -> ColorImage:
        return ColorImage(cv2.cvtColor(self.data * 255, cv2.COLOR_GRAY2BGR))

    def toGrayscale(self) -> GrayscaleImage:
        return GrayscaleImage(self.data * 255)

def otsuThreshold(image: GrayscaleImage) -> float:
    assert isinstance(image, GrayscaleImage)

    L = 256
    height, width = image.data.shape
    N = height * width
    n = image.histogram()
    p = n / N

    def ω(k: int) -> float:
        return sum(p[0:k])

    def μ(k: int) -> float:
        return sum([(i+1) * p_i for i, p_i in enumerate(p[0:k])])

    μ_T = μ(L)

    def σ_B(k: int) -> float: 
        numerator   = (μ_T * ω(k) - μ(k))**2
        denominator =  ω(k) * ( 1 - ω(k) )
        return numerator / denominator

    k = climb1dHill(list(range(L)), σ_B)

    return k

def climb1dHill(xs: List[int], evaluate: Callable[[int], Union[float, int]]) -> int:
    evaluations = {}

    def cachedEvaluate(index: int) -> float:
        if index not in evaluations:
            evaluations[index] = evaluate(index)
        return evaluations[index]

    def neighbors(index: int) -> Tuple[Optional[int], Optional[int]]:
        left, right = None, None
        if index > 0:
            left = index - 1
        if index < len(xs) - 1:
            right = index + 1
        return (left, right)

    startIndex = len(xs) // 2
    currentIndex, currentScore = startIndex, cachedEvaluate(xs[startIndex])

    while True:
        left, right = neighbors(currentIndex)
        if left is None or right is None:
            raise NotImplementedError

        leftScore, rightScore = cachedEvaluate(xs[left]), cachedEvaluate(xs[right])

        if left is not None and leftScore > currentScore:
            currentIndex, currentScore = left, leftScore
            continue

        if right is not None and rightScore > currentScore:
            currentIndex, currentScore = right, rightScore
            continue

        return xs[currentIndex]

def blur(image: GrayscaleImage, kernelSize: int = 2):

    def guassianKernel(size):
        return np.ones((size,size),np.float32) / (size**2)

    blurred = cv2.filter2D(image.data, -1, guassianKernel(kernelSize))
    return GrayscaleImage(blurred)

def otsuDetection(image: ColorImage, useBlur: bool = False, invert: bool = True) -> BinaryImage:
    greyscaleImage = image.toGrayscale()

    if useBlur:
        blurredImage = blur(greyscaleImage, kernelSize=3)
    else:
        blurredImage = greyscaleImage

    binaryImage = blurredImage.toBinary(inverse=invert)

    return binaryImage


def _denoise(image: BinaryImage, kernelSize: int = 3, erosions: int = 1, dilations: int = 1) -> BinaryImage:
    eroded = image.data

    for _ in range(erosions):
        eroded = cv2.erode(
            eroded,
            cv2.getStructuringElement(cv2.MORPH_CROSS, (kernelSize, kernelSize))
        )

    dilated = eroded

    for _ in range(dilations):
        dilated = cv2.dilate(
            dilated,
            cv2.getStructuringElement(cv2.MORPH_DILATE, (kernelSize, kernelSize))
        )

    return BinaryImage(dilated)

def shiftedPairs(signal: Union[List, np.ndarray], limit: Optional[int] = None) -> Iterator[Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    limit = len(signal) // 2 if limit is None else limit

    if limit > len(signal) // 2:
        raise ValueError("'limit' is greater than half the length of 'signal'")

    for offset in range(limit):
        if offset == 0:
            yield signal, signal
        else:
            yield signal[:-offset], signal[offset:]

def autocorrelation(signal: np.ndarray, limit: int = None) -> np.ndarray:
    return np.array([np.corrcoef(x, y)[0][1] for x, y in shiftedPairs(signal, limit)])

def _findFirstPeak(signal: np.ndarray, minHeight: float = 0.3, prominence: float = 0.05) -> Optional[int]:
    peaks, _ = scipy.signal.find_peaks(signal, prominence=prominence, height=minHeight)
    if len(peaks) == 0:
        return None
    else:
        return peaks[0]

def _estimateFirstPeakLocation(
    signal: np.ndarray,
    interpolate: bool = True,
    interpolationRadius: int = 2,
    interpolationGranularity: float = 0.01
) -> Optional[float]:
    assert interpolationRadius >= 1

    index = _findFirstPeak(signal)
    if index is None:
        return None

    if interpolate:
        start, end = index - interpolationRadius, index + interpolationRadius
        func = scipy.interpolate.interp1d(range(start, end + 1), signal[start:end + 1], kind='quadratic')
        newX = np.arange(start, end, interpolationGranularity)
        newY = func(newX)

        newPeak = newX[np.argmax(newY)]

        return newPeak

    else:
        return index
    
def _gridIsDetectable(image: BinaryImage) -> bool:
    columnDensity = np.sum(image.data, axis=0)
    columnFrequencyStrengths = autocorrelation(columnDensity)
    columnFrequency = _estimateFirstPeakLocation(
        columnFrequencyStrengths,
        interpolate=False
    )

    return not columnFrequency is None


def adaptive(image: ColorImage, applyDenoising: bool = False) -> BinaryImage:
    maxHedge = 1
    minHedge = 0.6  

    grayscaleImage = image.toGrayscale()
    otsuThreshold_value = otsuThreshold(grayscaleImage)

    hedging = float(maxHedge)
    binary = grayscaleImage.toBinary(otsuThreshold_value * hedging)

    while _gridIsDetectable(binary):
        hedging -= 0.05  
        if hedging < minHedge:
            break

        binary = grayscaleImage.toBinary(otsuThreshold_value * hedging)

    if applyDenoising:
        return _denoise(binary)
    else:
        return binary

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run signal detection on a folder of images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images (e.g., .png, .jpg)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save binary mask outputs")

    args = parser.parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.[pjPJ][pnPN]*[gG]"):
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"Could not read {img_file}, skipping.")
            continue

        color_img = ColorImage(img)
        binary_mask = adaptive(color_img)

        output_file = output_path / f"{img_file.stem}_mask.png"
        cv2.imwrite(str(output_file), binary_mask.data * 255)
        print(f"Saved: {output_file}")

        

