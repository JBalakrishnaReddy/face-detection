import argparse
import cv2
import dlib
import numpy
import os.path
import sys
import inspect
FACE_CLASSIFIER = CV_FACE_CLASSIFIER = '/home/bk/learning/projects/python/opencv/opencv-3.3.1/\
data/haarcascades/haarcascade_frontalface_alt.xml'
DLIB_FACE_CLASSIFIER = '/home/bk/Downloads/shape_predictor_68_face_landmarks.dat'
print(os.getcwd())
print (os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
# print(FACE_CLASSIFIER)
sys.exit()
ap = argparse.ArgumentParser()
ap.add_argument('-a', '--algorithm', required=False, help='You can define which methods library to use dlib or opencv')
ap.add_argument('-l', '--accuracy_level', required=False, help='Required level')
ap.add_argument('-i', '--image_path', required=True, help='Image path')

args = vars(ap.parse_args())
print(ap.parse_args(), args)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = numpy.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def use_cv(path, level=1.1):
    if os.path.isfile(path):
        face_classifiers = cv2.CascadeClassifier(FACE_CLASSIFIER)
        try:
            img = cv2.imread(path)
        except FileNotFoundError as err:
            print(err)
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifiers.detectMultiScale(gray, level)
        for i, (x,y,w,h) in enumerate(faces):
            if not os.path.isdir(os.getcwd()+'/ocv_files'):
                os.mkdir('ocv_files')
            current_path = os.getcwd()
            new_path = os.path.join(current_path, 'ocv_files')
            cv2.imwrite(new_path+'/img {}.png'.format(i), img[y:y+h, x:x+w])
            print('files saved successfully in path ', new_path+'/img {}.png'.format(i))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            img = cv2.putText(img, 'face #{}'.format(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        # print(x, y, w, h)
        # tr_img = img[y:y + h, x:x + w]
        # print(tr_img)
        # cv2.imshow(str(i), tr_img)
        im = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('img', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("file path is incorrect")


def use_dlib(path):
    if os.path.isfile(path):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(DLIB_FACE_CLASSIFIER)
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect)
            if not os.path.isdir(os.getcwd() + '/dlib_files'):
                os.mkdir('dlib_files')
            current_path = os.getcwd()
            new_path = os.path.join(current_path, 'dlib_files')
            cv2.imwrite(new_path+'/img {}.png'.format(i), image[y:y+h, x:x+w])

            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #
            # # show the face number
            # cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        im = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('img', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("file path is incorrect")


def main():
    if args['algorithm'] == 'dlib':
        # extract the images using dlib library
        use_dlib(args['image_path'])
    elif args['algorithm'] == 'opencv':
        use_cv(args['image_path'])
    else:
        print("available algorithms are \n1. dlib\n2. opencv\nPlease choose among them")


if __name__ == '__main__':
    main()
