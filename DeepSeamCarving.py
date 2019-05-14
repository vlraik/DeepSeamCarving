import numpy as np
import sys
from scipy.ndimage.filters import convolve
import time
from tqdm import trange
import cv2
import os


def calc_energy(img):
    """
    Calculates energy maps of the img
    :param img: input image
    :return: energy_map energy map of the image based on a specific energy function
    """
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    # print(convolved.shape)

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)
    # print(energy_map.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = np.uint8(gray_img)
    kernel = np.ones((5, 5), np.float32) / 25
    gray_img = cv2.medianBlur(gray_img, 5)
    energy_map = cv2.Canny(gray_img, 0, 0)
    cv2.imshow('energy', energy_map)

    key = cv2.waitKey(30) & 0xff
    return energy_map


def rectContains(rect, pt):
    """
    This function checks if the point is inside the bounding box

    :param rect: The bounding box of the object
    :param pt: Point to check it's position
    :return: True or False logic
    """
    logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    return logic


def comparePixels(image, boxes, compareArray, i, j, jend, idxs):
    """
    construct a function instead of argmin which can compare the pixels
    such that it also considers the weightage given from the deep learning algo
    this weightage will change when you compare these 3 points such that
    if a pixel is inside a bounding box then it is definitely not the minimum,
    then process the rest of the pixels.
    Thus, there are 3 possibilities:
    both the comparing pixels are inside the bounding box, or both are outside,
    in this case, use np.argmin to get further results.
    If one is inside and the other one is outside, compare only the pixel not in bounding
    box with the other pixel to check if it is lowest pixel

    Call the deep learning model function to get the rectangle on the image
    """

    idx = 0
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for index in idxs.flatten():
            # extract the bounding box coordinates
            # print(len(boxes))
            # print(idxs.flatten())
            # print(boxes)
            (x, y) = (boxes[index][0], boxes[index][1])
            (w, h) = (boxes[index][2], boxes[index][3])

            # draw a bounding box rectangle and label on the image
            # cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            inside = []
            for ycoord in range(j, jend + 1):
                inside.append(x < i < w and y < ycoord < h)

            if not all(inside):
                # print('yes')
                pass

            if all(inside) or not all(inside):
                idx = np.argmin(compareArray[i, j:jend])

            else:
                counter = 0
                smallest = 100000
                for ycoord in range(j, jend + 1):
                    if compareArray[i, ycoord] < smallest and inside[counter] == False:
                        smallest = compareArray[i, ycoord]
                        counter += 1
                    else:
                        counter += 1
                        continue
                idx = smallest

    return idx


def minimum_seam(img, net):
    """
    This is the pixel where the seams are sorted according to the energy map. Here we have to tweak this function and
    apply deep learning to get new weights for the seam
    :param img: Input image
    :return: Sorted array M, backtrack
    """
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    # img = cv2.resize(img, (100, 100))
    # cv2.imshow('image', img)
    # key = cv2.waitKey(30) & 0xff

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    # print(layerOutputs[0][0])
    end = time.time()

    # show timing information on YOLO
    #print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes, confidences, classIDs = get_boundingboxidxs(img, layerOutputs)

    # confidence and threshold
    confidencearg = sys.argv[5]
    thresholdarg = sys.argv[6]

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, float(confidencearg), float(thresholdarg))

    if (len(idxs.flatten() > 0)):
        for index in idxs.flatten():
            for i in range(1, r):
                for j in range(0, c):
                    (x, y) = (boxes[index][0], boxes[index][1])
                    (w, h) = (boxes[index][2], boxes[index][3])
                    # print(M[i,j])
                    # print(x,y,w,h)
                    if (x < i < w and y < j < h):
                        # print('f')
                        M[j, i] = 255
        cv2.imshow('M', M)

    key = cv2.waitKey(30) & 0xff

    for row in range(1, r):
        for column in range(0, c):
            # Handle the left edge of the image, to ensure we don't index -1
            if column == 0:

                idx = np.argmin(M[row - 1, column:column + 2])
                # print(idx)
                # idx = comparePixels(img, boxes, M, i - 1, j, j + 2, idxs)
                # print(idx)
                backtrack[row, column] = idx + column
                min_energy = M[row - 1, idx + column]
            else:
                idx = np.argmin(M[row - 1, column - 1:column + 2])
                # print(idx)
                # idx = comparePixels(img, boxes, M, i - 1, j - 1, j + 2, idxs)
                # print(idx)
                backtrack[row, column] = idx + column - 1
                min_energy = M[row - 1, idx + column - 1]

            M[row, column] += min_energy

    return M, backtrack


def carve_column(img, net):
    """
    Delete the seam with the lowest energy map.

    :param img: Input image
    :param net: Deep learning object
    :return: Carved image
    """
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img, net)

    # Create a (r, c) matrix filled with the value True
    # We'll be removing all pixels from the image which
    # have False later
    mask = np.ones((r, c), dtype=np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for count in reversed(range(r)):
        # Mark the pixels for deletion
        mask[count, j] = False
        j = backtrack[count, j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((r, c - 1, 3))

    return img


def crop_c(img, scale_c):
    """
    Removes specific number of seams from the image
    :param img: Input image
    :param scale_c: Scale of the image
    :return: output carved image
    """
    img = cv2.resize(img, (416, 416))
    r, c, _ = img.shape
    new_c = int(scale_c * c)

    # weight path and model configuration
    # weightsPath = sys.argv[6]
    # configPath = sys.argv[7]
    weightsPath = os.path.abspath("yolov3.weights")
    configPath = os.path.abspath("yolov3.cfg")

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    for index in trange(c - new_c):  # use range if you don't want to use tqdm
        img = carve_column(img, net)

    return img


def crop_r(img, scale_r):
    """
    Rotates the image to remove the seams
    :param img: Input image
    :param scale_r: Scale of the image
    :return: Output carved image
    """
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, scale_r)
    img = np.rot90(img, 3, (0, 1))
    return img


def get_boundingboxidxs(image, layerOutputs):
    """
    Generates the bounding box around the objects in the image
    :param image: Input image
    :param layerOutputs: The total numbers of boxes detected by the deep learning model.
    :return: Boxes, confidence levels of each box, and ClassIDs
    """
    # writing the deep learning code here with the object stored in a global variable
    # image = imread(in_filename)
    (H, W) = image.shape[:2]

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # confidence and threshold
    confidencearg = sys.argv[5]
    thresholdarg = sys.argv[6]
    labelsPath = os.path.abspath("coco-labels")
    LABELS = open(labelsPath).read().strip().split("\n")
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > float(confidencearg):
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidencearg, thresholdarg)

    return boxes, confidences, classIDs


def main():
    """
    Add more arguments from deep learning model or hard code them
    :arg: 1. Which axis row/column
    :arg: 2. Scale to be used, for example, reduce the size of image to half would be 0.5
    :arg: 3. Input image
    :arg: 4. Output image
    :arg: 5. Confidence value, default put as 0.5
    :arg: 6. Threshold value, default put as 0.3
    :return:
    """

    if len(sys.argv) != 7:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    which_axis = sys.argv[1]
    scale = float(sys.argv[2])
    in_filename = sys.argv[3]
    out_filename = sys.argv[4]

    image = cv2.imread(in_filename)
    if which_axis == 'r':
        out = crop_r(image, scale)
    elif which_axis == 'c':
        out = crop_c(image, scale)
    else:
        print('usage: carver.py <r/c> <scale> <image_in> <image_out>', file=sys.stderr)
        sys.exit(1)

    cv2.imwrite(out_filename, out)


"""
python carver.py r 0.5 image2.jpg cropped.jpg
Run the function with such argument style
"""
if __name__ == '__main__':
    main()
