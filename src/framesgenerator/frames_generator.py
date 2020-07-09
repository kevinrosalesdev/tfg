import os
import cv2
import re
import numpy

from PIL import Image


def generate_frames(filename):
    print("[Getting frames from video '" + os.getenv('VIDEOS_ROUTE') + "/" + filename + "']")
    directory = os.getenv('MAIN_ROUTE') + "/out/frames-generator/" + os.path.splitext(filename.replace("/", "-"))[0]
    used_camera = int(re.findall('\d+', filename)[5])
    if used_camera == 3 or used_camera == 6:
        deinterlace_frame = True
    else:
        deinterlace_frame = False

    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            print("[Results folder already exists. Existing frames will not be updated]")
            return 0

    except OSError:
        print('[Error creating frames folder]')
        return -1

    video = cv2.VideoCapture(os.getenv('VIDEOS_ROUTE') + "/" + filename)
    count = 1
    while True:
        state, image = video.read()
        if not state:
            break

        if deinterlace_frame:
            image = Image.fromarray(image)
            size = list(image.size)
            image = image.resize([size[0], int(size[1]/2)], Image.NEAREST)
            image = numpy.array(image.resize(size))

        cv2.imwrite(directory + "/frame-" + str(count).zfill(3) + ".jpg", image)
        count += 1

    video.release()
    print("[Video frames have been generated in '" + directory + "']")
    return 0
