# face-detection
Packages used
1. opencv -- used for image processing
2. numpy  -- numerical array package
3. dlib   -- used for image processing

What the script does?
It will detect the faces in the given image according to the algorithm choosen and save those images in local directories
'ocv files' or 'dlib files'

How to execute the scipt:
1. Change your executing directory to the actual source tree.
2. Open the terminal and execute the below script.

	python3 src/main.py --method $aaaa --image_path $your_image_path --show_output $True/False

Available methods are
1. dlib
2. opencv

Eg: python3 main_v3.py --algorithm opencv --image_path /home/$user_name/Pictures/test.png


Solution to actual problem: Algorithm
1. Opencv and dlib packages/method are used for object/face detection. 
2. In one of the method "dlib" was used to detect in faces directly. A small tweek was necessary in order to capture the faces even if though faces are not very clear but still they are detectable faces. Tweeking it more ruins the face detection.
3. In another method "Opencv" was used to detect the faces by using the pre-trained haarcascades. This is how the faces in the given picture are detected.
4. Once the faces are detected i.e once we extract the coordinates of polygons surronding the face we go to next step.
5. In this step the coordinates extracted above are used to crop the picture of face after which the actual size of the image is cropped "10%" by removing (0.5 * width) and (0.5 * height) on left and right, top and bottom. The resulting image is approximately 81% of the actual detected face. List slicing was used to crop the image. 
6. Once the images are cropped to 81% they are saved in a directory named "ocv_files" or "dlib_files" depending on the method choosen whie starting the script.

