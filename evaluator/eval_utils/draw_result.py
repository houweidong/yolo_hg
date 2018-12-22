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
        # cv2.rectangle(img, (x - w, y - h - 20),
        #               (x + w, y - h), (125, 125, 125), -1)
        # lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
        # cv2.putText(
        #     img, result[i][0] + ' : %.2f' % result[i][5],
        #     (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #     (0, 0, 0), 1, lineType)