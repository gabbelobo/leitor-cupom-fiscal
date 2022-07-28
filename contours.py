import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import re
import pandas as pd
import re
from PIL import Image

def find_amounts(text):
    amounts = re.findall(r'\d+[\.,\,]{1}\d{2}\b', text)
    floats = [float(amount.replace(",", ".")) for amount in amounts]
    unique = list(dict.fromkeys(floats))
    return unique
def build_df(text):
  rows = text.split('\n')

  indices_valores = [i for i, x in enumerate(rows) if (x.endswith('¢')  or len(re.findall("^\d+ U[NM] X", x)) != 0)]
  indices_descontos = [i for i, x in enumerate(rows) if (x.startswith('desconto'))]
  
  numbers = np.array([])
  names = np.array([])
  values = np.array([])

  for i in indices_valores:
    row_tries = 3
    for k in range(row_tries):
        name_groups = re.findall("([a-zA-Z0-9]+) [\d{14} O)]+ (.+$)", rows[i-1 - k])
        if(len(name_groups) > 0):
            break
    name_groups = name_groups[0]
    price = re.search("(\d+\,\s?\d{2}¢)$", rows[i])
    if(price is None):
        price = re.findall("^\d+ U[NM] X (\d+,\d+)", rows[i])[0]
    else:
        price = price[0][:-1]

    numbers = np.append(numbers, i-1)
    names = np.append(names, name_groups[1])
    values = np.append(values, float(price.replace(' ', '').replace(',', '.')))
  
  descontos = np.zeros(values.size)

  for i in indices_descontos:
    desconto_groups = re.findall("^desconto ite[mn] (\d+) [-~](\d\,\d{2})", rows[i])[0]
    index = int(desconto_groups[0]) - 1
    desconto = float(desconto_groups[1].replace(',', '.'))
    descontos[index] = desconto
  d = {
      'Number': numbers, 
      'Name': names,
      'Value': values,
      'Desconto': descontos,
      'Total': values - descontos
       }
  df = pd.DataFrame(data=d)

  return df

def contour_to_rect(contour):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def wrap_perspective(img, rect):
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    
def bw_scanner(image, offset):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _,threshed = cv2.threshold(gray,130+offset,255,cv2.THRESH_BINARY)
    print(130+offset)
    output = Image.fromarray(threshed)
    output.save('result.png')
    # T = threshold_local(gray, 21, offset = 5, method = "gaussian")
    return threshed

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    # return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return cv2.resize(image, dim)

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
def plot_gray(image):
    plt.figure(figsize=(16,10))
    return plt.imshow(image, cmap='Greys_r')

def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    result = None
    for i in range(32):
        result = cv2.approxPolyDP(contour, (0.001*i + 0.010) * peri, True)
        if result is not None:
            break
    return result
    # return cv2.approxPolyDP(contour, 0.02 * peri, True)

def get_receipt_contour(contours):    
  for c in contours:
      approx = approximate_contour(c)
      if len(approx) == 4:
          return approx

def get_contours(image):

    # converting to LAB color space
    # lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # l_channel, a, b = cv2.split(lab)

    # # Applying CLAHE to L-channel
    # # feel free to try different values for the limit and grid size:
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # cl = clahe.apply(l_channel)

    # # merge the CLAHE enhanced L-channel with the a and b channel
    # limg = cv2.merge((cl,a,b))

    # # Converting image from LAB Color model to BGR color spcae
    # enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    resize_ratio = 500 / image.shape[0]
    # resize_ratio = 0.2
    image_resized = opencv_resize(image, resize_ratio)

    img_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    dilated = cv2.dilate(blurred, rectKernel)

    edged = cv2.Canny(dilated, 50, 100)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)

    receipt_contour = get_receipt_contour(largest_contours)

    return receipt_contour / resize_ratio


def get_csv(image, contours):
    scanned = wrap_perspective(image.copy(), contour_to_rect(contours))
    for i in range(8):
        result = bw_scanner(scanned, i*2)
        extracted_text = pytesseract.image_to_string(result)
        # print(extracted_text)
        try:
            df = build_df(extracted_text)
            return df
        except:
            continue
    
