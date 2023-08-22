import os
import numpy as np

# Function to calculate Intersection over Union given the inferences and ground truth
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(x1 - w1 / 2, x2 - w2 / 2)
    y_intersection = max(y1 - h1 / 2, y2 - h2 / 2)
    x2_intersection = min(x1 + w1 / 2, x2 + w2 / 2)
    y2_intersection = min(y1 + h1 / 2, y2 + h2 / 2)

    # Calculate the area of intersection
    intersection_area = max(0, x2_intersection - x_intersection) * max(0, y2_intersection - y_intersection)

    # Calculate the area of the two bounding boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate IoU
    iou = intersection_area / (area1 + area2 - intersection_area)

    return iou

# A function to convert the inference text file to usable format
def txt_file_to_list(input_path):
    file = open(input_path, "r+")
    string_list = file.read()
    string_list = string_list.split('\n')

    array_list = []
    for line in string_list:
        if line.strip():  # To prevent empty lines into making into the array
            # Split the current input string by spaces and convert each element to a float
            float_array = [float(x) for x in line.split()]
            # Append the float array to the list of float_arrays
            array_list.append(float_array)

    return array_list

ground_truth_folder = r"D:\Assignments\Achira_Labs\Model_Training\test\labels"

inference_folder = r"D:\Assignments\Achira_Labs\Code\runs\detect\predict\labels"
inference_files = os.listdir(inference_folder)

confidence_threshold = 0.98  # Confidence of the inference
IOU_threshold = 0.95  # Minimum IOU to consider a good inference

number_of_classes = 4  # This parameter could be acquired through user input if the script has general usage.

TPs = np.zeros(number_of_classes)
FPs = np.zeros(number_of_classes)
FNs = np.zeros(number_of_classes)

for file in inference_files:

    inferences = txt_file_to_list(os.path.join(inference_folder, file))
    grounds = txt_file_to_list(os.path.join(ground_truth_folder, file))

    iou_map = np.ones(len(inferences)) * 99999999
    iou_index = np.ones(len(inference_files)) * 99999999

    for iter_inference in range(len(inferences)):
        if inferences[iter_inference][-1] > confidence_threshold:  # Check if the inference is above the confidence threshold. It is a False positive if it isn't.
            best_iou = 0
            best_gt_index = None

            for iter_gt in range(len(grounds)):  # Mapping the inference to the ground truth
                if inferences[iter_inference][0] == grounds[iter_gt][0]:
                    temp_iou = calculate_iou(inferences[iter_inference][1:5], grounds[iter_gt][1:])

                    if temp_iou > best_iou:
                        best_iou = temp_iou
                        best_gt_index = iter_gt

            if best_iou > IOU_threshold:  # Check if the inference is above the IOU threshold. It is a false positive if it isn't.

                if best_gt_index in iou_index:  #If two inferences mapped to the same ground truth, then one of it is a false positive.
                    FPs[int(inferences[iter_inference][0])] += 1

                else:
                    TPs[int(inferences[iter_inference][0])] += 1

                iou_map[iter_inference] = best_iou
                iou_index[iter_inference] = best_gt_index
            else:
                iou_map[iter_inference] = None
                iou_index[iter_inference] = best_gt_index

                FPs[int(inferences[iter_inference][0])] += 1
        else:
            iou_map[iter_inference] = None
            iou_index[iter_inference] = None

            FPs[int(inferences[iter_inference][0])] += 1

    for iter_grounds in range(len(grounds)):
        if iter_grounds not in iou_index:
            FNs[int(grounds[iter_grounds][0])] += 1


precision = TPs / (TPs + FPs)
recall = TPs / (TPs + FNs)
F1_score = 2 * precision * recall / (precision + recall)

# print('At the given confidence threshold of ' + str(confidence_threshold) + ' and IOU threshold of ' + str(IOU_threshold) + ', the F1_score is ' + str(round(F1_score), 2))