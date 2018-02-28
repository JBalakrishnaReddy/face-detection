import argparse
import os.path
import numpy
import dlib
import cv2
import sys

CV_FACE_CLASSIFIER = os.getcwd() + '/haarcascades/haarcascade_frontalface_alt.xml'
# FACE_CLASSIFIER = CV_FACE_CLASSIFIER = '/home/bk/learning/projects/python/opencv/opencv-3.3.1/\
# data/haarcascades/haarcascade_frontalface_alt.xml'
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


def im_transparent(image, x, y, w, h):
    im2 = image[y:y + h, x:x + w]
    mask = numpy.zeros(im2.shape[:2], numpy.uint8)
    bgdModel = numpy.zeros((1, 65), numpy.float64)
    fgdModel = numpy.zeros((1, 65), numpy.float64)
    rect = (1, 1, w, h)
    cv2.grabCut(im2, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    im2 = im2 * mask2[:, :, numpy.newaxis]
    
    im3, width, height = shrink_n_border(image, x, y, h, w)
    #cv2.imshow('before cut', im3)
    mask_3 = numpy.zeros(im3.shape[:2], numpy.uint8)
    bgdModel = numpy.zeros((1, 65), numpy.float64)
    fgdModel = numpy.zeros((1, 65), numpy.float64)
    rect = (1, 1, im3.shape[1], im3.shape[0])# w, h)
    cv2.grabCut(im3, mask_3, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = numpy.where((mask_3 == 2) | (mask_3 == 0), 0, 1).astype('uint8')
    
    im3 = im3 * mask2[:, :, numpy.newaxis]    
    height = round((im2.shape[0] - im3.shape[0])/2)
    width = round((im2.shape[1] - im3.shape[1])/2)
    im3 = cv2.copyMakeBorder(im3, height, im2.shape[0] - im3.shape[0] - height, width,
                            im2.shape[1] - im3.shape[1] - width,
                            cv2.BORDER_CONSTANT, value=[0, 0, 0])
    _, im4 = cv2.threshold(cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY)
    r_channel, g_channel, b_channel = cv2.split(im2)
    a_channel = numpy.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    actual_RGBA = cv2.merge((r_channel, g_channel, b_channel, a_channel))
    r_channel, g_channel, b_channel = cv2.split(cv2.bitwise_and(cv2.cvtColor(im4, cv2.COLOR_GRAY2BGR), im2))
    a_channel = numpy.where((r_channel > 0) | (g_channel > 0) | (b_channel > 0), 255, 0).astype('uint8')
    shrinked_RGBA = cv2.merge((r_channel, g_channel, b_channel, a_channel))    
    return actual_RGBA, shrinked_RGBA


def shrink_n_border(img, x, y, w, h):
    width = int(round(0.05 * w))
    height = int(round(0.05 * h))
    img = img[y:y+h, x:x+w]
    im = cv2.resize(img, (0,0), fx=0.9, fy=0.9)
    # return im, w-2*width, h-2*height
    # im = cv2.copyMakeBorder(im, height, img.shape[0]-im.shape[0]-height, width, img.shape[0]-im.shape[0]-width, cv2.BORDER_CONSTANT,value=[255,0,0])
    return im, width, height


def shrink_n_save(img, i, x, y, w, h, path):
    width = int(round(0.05 * w))
    height = int(round(0.05 * h))
    cv2.imwrite(path+'/img {}.png'.format(13 + i), img[y+height:y+h-height, x+width:x+w-width])


def use_cv(classifier, path, level=1.1, show=True):
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
            ret, ret1 = im_transparent(image, x, y, w, h)
            cv2.imwrite(new_path + '/actual img {}.png'.format(i), ret)
            cv2.imwrite(new_path + '/shrink img {}.png'.format(i), ret1)
            # shrink_n_save(image, i, x, y, w, h, new_path)
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
            # shrink_n_save(image, i, x, y, w, h, new_path)
            ret, ret1 = im_transparent(image, x, y, w, h)
            cv2.imwrite(new_path + '/actual img {}.png'.format(i), ret)
            cv2.imwrite(new_path + '/shrink img {}.png'.format(i), ret1)

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
