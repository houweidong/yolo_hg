import cv2


def draw_result(img, result, option=(0, 255, 0)):
    for i in range(len(result)):
        # print(result[i][0])
        # if result[i][0] != 'cat':
        #    continue
        x = int(result[i][0])
        y = int(result[i][1])
        w = int(result[i][2] / 2)
        h = int(result[i][3] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), option, 2)
