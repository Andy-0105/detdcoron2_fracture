import os,cv2
path="/home/j2031/detectron2/pre_out"

def read_path(file_pathname):
    for filename in os.listdir(file_pathname):
        print(filename)
        img = cv2.imread(file_pathname+'/'+filename)
        ret, th=cv2.threshold(img,0.5,255,cv2.THRESH_BINARY)
        cv2.imwrite(f'/home/j2031/detectron2/mask/{filename}',th)
read_path(path)