import argparse
import os.path
import numpy
import dlib
import cv2

CV_FACE_CLASSIFIER = os.getcwd() + '/haarcascades/haarcascade_frontalface_alt.xml'
# print(CV_FACE_CLASSIFIER)
# sys.exit()

ap = argparse.ArgumentParser()
ap.add_argument('-a', '--method', required=True, help='You can define which \
methods library to use dlib or opencv')
ap.add_argument('-i', '--image_path', required=True, help='Image path')
ap.add_argument('-s', '--show_output', required=False, help='If you want to \
see the detected faces')
args = vars(ap.parse_args())


def shape_to_np(shape, dtype="int"):
    coords = numpy.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h


def shrink_n_save(img, i, x, y, w, h, path):
    width = int(round(0.05 * w))
    height = int(round(0.05 * h))
    cv2.imwrite(path+'/img {}.png'.format(13 + i), img[y+height:y+h-height, x+width:x+w-width])


def use_cv(classifier, path, level=1.1, show=False):
    if os.path.isfile(path):
        face_classifiers = cv2.CascadeClassifier(classifier)
        try:
            image = cv2.imread(path)
        except FileNotFoundError as err:
            print(err)
            return
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifiers.detectMultiScale(gray, level)
        for i, (x, y, w, h) in enumerate(faces):
            if not os.path.isdir(os.getcwd()+'/ocv_files'):
                os.mkdir('ocv_files')
            current_path = os.getcwd()
            new_path = os.path.join(current_path, 'ocv_files')
            shrink_n_save(image, i, x, y, w, h, new_path)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, 'face #{}'.format(i), (x, y - 5),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        if show == True:
            im = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('img', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("file path is incorrect")


def use_dlib(path, show=False):
    if os.path.isfile(path):
        detector = dlib.get_frontal_face_detector()
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 2)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect)
            if not os.path.isdir(os.getcwd() + '/dlib_files'):
                os.mkdir('dlib_files')
            current_path = os.getcwd()
            new_path = os.path.join(current_path, 'dlib_files')
            shrink_n_save(image, i, x, y, w, h, new_path)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),\
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if show == True:
            im = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('img', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("file path is incorrect")


def main():
    if args['method'] == 'dlib':
        # extract the images using dlib library
        use_dlib(args['image_path'],\
            show=True if args['show_output'] == 'True' else False)
    elif args['method'] == 'opencv':
        use_cv(CV_FACE_CLASSIFIER, args['image_path'],\
            show=True if args['show_output'] == 'True' else False)
    else:
        print("available methods are \n1. dlib\n2. opencv\nPlease choose among them")


if __name__ == '__main__':
    main()
