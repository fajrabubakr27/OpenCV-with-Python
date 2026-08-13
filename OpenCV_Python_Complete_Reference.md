# OpenCV with Python - Complete Reference Guide

> A comprehensive, code-first documentation based on the [OpenCV Course - Full Tutorial with Python](https://www.youtube.com/watch?v=oXlwWbU8l2o).  
> Use this as your primary reference so you never have to re-watch the course.

---

## Table of Contents

- [Installation](#installation)
- [Section 1: Basics](#section-1-basics)
  - [Reading Images & Video](#reading-images--video)
  - [Resizing and Rescaling Frames](#resizing-and-rescaling-frames)
  - [Drawing Shapes & Putting Text](#drawing-shapes--putting-text)
  - [5 Essential Functions in OpenCV](#5-essential-functions-in-opencv)
  - [Image Transformations](#image-transformations)
  - [Contour Detection](#contour-detection)
- [Section 2: Advanced](#section-2-advanced)
  - [Color Spaces](#color-spaces)
  - [Color Channels](#color-channels)
  - [Blurring](#blurring)
  - [Bitwise Operations](#bitwise-operations)
  - [Masking](#masking)
  - [Histogram Computation](#histogram-computation)
  - [Thresholding / Binarizing Images](#thresholding--binarizing-images)
  - [Edge Detection](#edge-detection)
- [Section 3: Faces](#section-3-faces)
  - [Face Detection with Haar Cascades](#face-detection-with-haar-cascades)
  - [Face Recognition with OpenCV's Built-in Recognizer](#face-recognition-with-opencvs-built-in-recognizer)
- [Quick Reference Cheat Sheet](#quick-reference-cheat-sheet)

---

## Installation

```bash
pip install opencv-python opencv-contrib-python
```

> `opencv-contrib-python` is required for extra modules like `face` (LBPH recognizer).

```python
import cv2 as cv
print(cv.__version__)
```

---

## Section 1: Basics

### Reading Images & Video

#### Read an Image
```python
import cv2 as cv

img = cv.imread('../Resources/Photos/cats.jpg')
cv.imshow('Cats', img)
cv.waitKey(0)  # waits indefinitely for a key press
```

#### Read a Video
```python
capture = cv.VideoCapture('../Resources/Videos/dog.mp4')

while True:
    isTrue, frame = capture.read()

    if isTrue:
        cv.imshow('Video', frame)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break
    else:
        break

capture.release()
cv.destroyAllWindows()
```

**Key Notes:**
- `cv.imread(path)` → returns a NumPy array (BGR format).
- `cv.VideoCapture(0)` → use `0` for webcam.
- Always `release()` the capture and `destroyAllWindows()` to free memory.

---

### Resizing and Rescaling Frames

#### Generic Rescale Function (Images, Videos, Live Video)
```python
def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)
```

#### Change Resolution for Live Video Only
```python
def changeRes(width, height):
    capture.set(3, width)   # 3 = width property
    capture.set(4, height)  # 4 = height property
```

#### Resize with Specific Interpolation
```python
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)
```

| Interpolation | Best For |
|---------------|----------|
| `cv.INTER_AREA` | Downscaling (shrinking) |
| `cv.INTER_LINEAR` | Default, fast |
| `cv.INTER_CUBIC` | Upscaling (slower, higher quality) |

---

### Drawing Shapes & Putting Text

```python
import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')

# 1. Paint a region
blank[200:300, 300:400] = 0, 0, 255  # BGR: Red
cv.imshow('Green', blank)

# 2. Rectangle
# cv.rectangle(img, pt1, pt2, color, thickness)
# thickness=-1 fills the shape
cv.rectangle(blank, (0, 0), (blank.shape[1]//2, blank.shape[0]//2), (0, 255, 0), thickness=-1)

# 3. Circle
# cv.circle(img, center, radius, color, thickness)
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0, 0, 255), thickness=-1)

# 4. Line
cv.line(blank, (100, 250), (300, 400), (255, 255, 255), thickness=3)

# 5. Text
cv.putText(blank, 'Hello, my name is Jason!!!', (0, 225),
           cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)

cv.imshow('Text', blank)
cv.waitKey(0)
```

---

### 5 Essential Functions in OpenCV

```python
import cv2 as cv

img = cv.imread('../Resources/Photos/park.jpg')

# 1. Convert to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 2. Blur (Gaussian)
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)

# 3. Edge Cascade (Canny)
canny = cv.Canny(blur, 125, 175)

# 4. Dilating (thicken edges)
dilated = cv.dilate(canny, (7, 7), iterations=3)

# 5. Eroding (thin edges back)
eroded = cv.erode(dilated, (7, 7), iterations=3)

# Bonus: Resize & Crop
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cropped = img[50:200, 200:400]  # array slicing [y1:y2, x1:x2]
```

| Function | Purpose |
|----------|---------|
| `cv.cvtColor` | Color space conversion |
| `cv.GaussianBlur` | Noise reduction |
| `cv.Canny` | Edge detection |
| `cv.dilate` | Expand white regions |
| `cv.erode` | Shrink white regions |

---

### Image Transformations

```python
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/park.jpg')

# --- Translation ---
def translate(img, x, y):
    transMat = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, transMat, dimensions)

# -x → Left, -y → Up, x → Right, y → Down
translated = translate(img, -100, 100)

# --- Rotation ---
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, -45)

# --- Flipping ---
# 0 = vertical, 1 = horizontal, -1 = both
flip = cv.flip(img, -1)

# --- Cropping ---
cropped = img[200:400, 300:400]
```

---

### Contour Detection

Contours are the boundaries of shapes. Usually detected on **Canny edges** or **thresholded images**.

```python
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
blank = np.zeros(img.shape, dtype='uint8')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)

# Find contours
contours, hierarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')

# Draw contours on blank canvas
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)
```

**Contour Retrieval Modes:**
- `cv.RETR_LIST` — all contours, no hierarchy
- `cv.RETR_EXTERNAL` — only external contours
- `cv.RETR_TREE` — full hierarchy

**Approximation Methods:**
- `cv.CHAIN_APPROX_SIMPLE` — compresses horizontal, vertical, diagonal segments
- `cv.CHAIN_APPROX_NONE` — stores all contour points

---

## Section 2: Advanced

### Color Spaces

```python
import cv2 as cv

img = cv.imread('../Resources/Photos/park.jpg')

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# BGR to HSV (Hue, Saturation, Value)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

# BGR to RGB (for matplotlib)
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Convert back: LAB → BGR
lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
```

> **Important:** OpenCV reads images in **BGR** by default. Matplotlib expects **RGB**.

---

### Color Channels

```python
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/park.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

# Split channels
b, g, r = cv.split(img)

# Display individual channels in color
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

# Merge back
merged = cv.merge([b, g, r])
```

---

### Blurring

Blurring reduces high-frequency noise.

```python
import cv2 as cv

img = cv.imread('../Resources/Photos/cats.jpg')

# 1. Averaging (simple mean of kernel)
average = cv.blur(img, (3, 3))

# 2. Gaussian Blur (weighted mean, natural look)
gauss = cv.GaussianBlur(img, (3, 3), 0)

# 3. Median Blur (great for salt-and-pepper noise)
median = cv.medianBlur(img, 3)  # kernel size must be odd

# 4. Bilateral Filter (preserves edges while smoothing)
bilateral = cv.bilateralFilter(img, 10, 35, 25)
# params: diameter, sigmaColor, sigmaSpace
```

| Method | Best Use Case |
|--------|---------------|
| Averaging | General smoothing |
| Gaussian | Natural blur, pre-processing |
| Median | Salt & pepper noise removal |
| Bilateral | Edge-preserving smoothing |

---

### Bitwise Operations

Used heavily in **masking**.

```python
import cv2 as cv
import numpy as np

blank = np.zeros((400, 400), dtype='uint8')
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

# AND → intersection only
bitwise_and = cv.bitwise_and(rectangle, circle)

# OR → union (both regions)
bitwise_or = cv.bitwise_or(rectangle, circle)

# XOR → non-intersecting regions
bitwise_xor = cv.bitwise_xor(rectangle, circle)

# NOT → invert
bitwise_not = cv.bitwise_not(circle)
```

---

### Masking

Masking allows you to focus on a specific region of interest (ROI).

```python
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/cats 2.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

# Create shapes for the mask
circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45, img.shape[0]//2), 100, 255, -1)
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)

# Combine shapes with bitwise AND
weird_shape = cv.bitwise_and(circle, rectangle)

# Apply mask to image
masked = cv.bitwise_and(img, img, mask=weird_shape)
cv.imshow('Masked Image', masked)
```

> The `mask` parameter must be a **single-channel (grayscale)** image where white = keep, black = ignore.

---

### Histogram Computation

Histograms show pixel intensity distribution.

```python
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('../Resources/Photos/cats.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

# Create a circular mask
mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=mask)

# Compute color histogram
plt.figure()
plt.title('Colour Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')

for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()
```

**Grayscale Histogram (commented out in original):**
```python
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_hist = cv.calcHist([gray], [0], mask, [256], [0, 256])
```

| Parameter | Meaning |
|-----------|---------|
| `[img]` | List of source images |
| `[i]` | Channel index (0=B, 1=G, 2=R) |
| `mask` | Optional mask (None for full image) |
| `[256]` | Number of bins |
| `[0, 256]` | Pixel value range |

---

### Thresholding / Binarizing Images

Converts grayscale to binary (black & white) based on a threshold.

```python
import cv2 as cv

img = cv.imread('../Resources/Photos/cats.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# --- Simple Thresholding ---
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)

# --- Adaptive Thresholding ---
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    11,   # block size (odd number)
    9     # constant subtracted from mean
)
```

| Type | Description |
|------|-------------|
| `THRESH_BINARY` | Pixel > thresh → max, else → 0 |
| `THRESH_BINARY_INV` | Inverse of above |
| `ADAPTIVE_THRESH_MEAN_C` | Threshold = mean of neighborhood |
| `ADAPTIVE_THRESH_GAUSSIAN_C` | Threshold = Gaussian-weighted sum |

> Adaptive thresholding is best for images with **varying lighting conditions**.

---

### Edge Detection

#### Canny Edge Detector
```python
canny = cv.Canny(gray, 150, 175)  # threshold1, threshold2
```

#### Laplacian & Sobel (Gradient-based)
```python
import cv2 as cv
import numpy as np

img = cv.imread('../Resources/Photos/park.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Laplacian (2nd derivative)
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))

# Sobel (1st derivative, directional)
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)  # x-direction
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)  # y-direction
combined_sobel = cv.bitwise_or(sobelx, sobely)
```

| Method | How It Works |
|--------|--------------|
| **Canny** | Multi-stage: noise reduction → gradient → non-max suppression → hysteresis thresholding |
| **Laplacian** | Detects rapid intensity change (zero-crossing) |
| **Sobel** | Computes gradient in X and Y directions separately |

---

## Section 3: Faces

### Face Detection with Haar Cascades

Haar Cascades are pre-trained classifiers stored in XML files.

```python
import cv2 as cv

img = cv.imread('../Resources/Photos/group 1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Load the classifier
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Detect faces
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

# Draw rectangles
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Faces', img)
cv.waitKey(0)
```

| Parameter | Description |
|-----------|-------------|
| `scaleFactor` | How much the image size is reduced at each scale (e.g., 1.1 = 10% reduction) |
| `minNeighbors` | How many neighbors each candidate rectangle should have to retain it (higher = fewer false positives) |

---

### Face Recognition with OpenCV's Built-in Recognizer

This uses **LBPH (Local Binary Patterns Histograms)** face recognizer.

#### Step 1: Train the Model

```python
import os
import cv2 as cv
import numpy as np

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = r'..Media Files\Faces\train'

haar_cascade = cv.CascadeClassifier('haar_face.xml')
features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)

            if img_array is None:
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ---------------')

features = np.array(features, dtype='object')
labels = np.array(labels)

# Create and train recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features, labels)

# Save model
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)
```

#### Step 2: Recognize Faces

```python
import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

# Load trained model
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

# Load test image
img = cv.imread(r'../Resources\Faces\val\elton_john/1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect face
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]

    # Predict
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    # Display result
    cv.putText(img, str(people[label]), (20, 20),
               cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
```

> **Lower confidence value = better match** (it's essentially a distance metric).

---

## Quick Reference Cheat Sheet

### Core I/O
| Code | Action |
|------|--------|
| `cv.imread(path)` | Read image |
| `cv.imshow(name, img)` | Show image |
| `cv.waitKey(0)` | Wait for key (0 = infinite) |
| `cv.VideoCapture(path/0)` | Capture video / webcam |
| `cv.cvtColor(img, code)` | Convert color space |

### Shape & Transform
| Code | Action |
|------|--------|
| `cv.resize(img, dim, interp)` | Resize image |
| `cv.flip(img, code)` | Flip image (0/1/-1) |
| `cv.warpAffine(img, M, dim)` | Affine transform |
| `cv.getRotationMatrix2D(...)` | Get rotation matrix |
| `img[y1:y2, x1:x2]` | Crop with NumPy slicing |

### Drawing
| Code | Action |
|------|--------|
| `cv.rectangle(img, p1, p2, color, thick)` | Draw rectangle |
| `cv.circle(img, center, r, color, thick)` | Draw circle |
| `cv.line(img, p1, p2, color, thick)` | Draw line |
| `cv.putText(img, text, origin, font, scale, color, thick)` | Put text |

### Processing
| Code | Action |
|------|--------|
| `cv.GaussianBlur(img, ksize, sigma)` | Gaussian blur |
| `cv.Canny(img, t1, t2)` | Edge detection |
| `cv.dilate(img, kernel, iter)` | Dilate |
| `cv.erode(img, kernel, iter)` | Erode |
| `cv.threshold(img, thresh, max, type)` | Simple threshold |
| `cv.adaptiveThreshold(...)` | Adaptive threshold |

### Contours
| Code | Action |
|------|--------|
| `cv.findContours(img, mode, method)` | Find contours |
| `cv.drawContours(img, contours, idx, color, thick)` | Draw contours (-1 = all) |

### Bitwise & Masking
| Code | Action |
|------|--------|
| `cv.bitwise_and(a, b, mask=?)` | AND |
| `cv.bitwise_or(a, b)` | OR |
| `cv.bitwise_xor(a, b)` | XOR |
| `cv.bitwise_not(a)` | NOT |
| `cv.bitwise_and(img, img, mask=mask)` | Apply mask |

### Histogram
| Code | Action |
|------|--------|
| `cv.calcHist([img], [ch], mask, bins, range)` | Compute histogram |

### Face
| Code | Action |
|------|--------|
| `cv.CascadeClassifier(xml)` | Load Haar cascade |
| `classifier.detectMultiScale(gray, ...)` | Detect objects |
| `cv.face.LBPHFaceRecognizer_create()` | Create recognizer |
| `recognizer.train(features, labels)` | Train model |
| `recognizer.read(path)` / `save(path)` | Load / Save model |
| `recognizer.predict(roi)` | Predict label & confidence |

---
