# face-detection
Packages used
1. opencv -- used for image processing
2. numpy  -- numerical array package
3. dlib   -- used for image processing

What the script does?
It will detect the faces in the given image according to the algorithm choosen and save those images in local directories
'ocv files' or 'dlib files'

How to execute the scipt:
python3 main_vxx.py --algorithm $aaaa --image_path $your_image_path
Available algorithms are 
1. dlib
2. opencv

Eg: python3 main_v3.py --algorithm opencv --image_path /home/$user_name/Pictures/test.png

