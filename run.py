from  tqdm import tqdm
import cv2
from source.cartoonize import Cartoonizer
import os
import numpy as np

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def concat(pa):
    filenames = [x for x in os.listdir(pa) if os.path.isfile(os.path.join(pa,x))  ]
    c1 = len(filenames)
    print(filenames)
    i = 0
    print(c1)

    img1 = cv2.imread('picture/pic/07.jpg')
    h, w, c = img1.shape
    cut1 = np.ones((h, 7, 3), dtype='u8') * 255
    print(img1.shape, cut1.shape)
    im_A = np.concatenate([img1, cut1], 1)
    # im_A = np.hstack([img1, cut1])
    for filepath in tqdm(filenames):
        i += 1
        img_path = os.path.join(pa, filepath)
        img = cv2.imread(img_path)
        cv2.putText(img, f"{filepath[:-4]}", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
        print(img.shape)
        # cut1 = np.ones((5, w, 3), dtype='u8') * 255
        im_A = np.concatenate([im_A, cut1, img], 1)
    cv2.imwrite(os.path.join(pa ,  'res.jpg'), im_A)

models = {
    "anime": Cartoonizer(dataroot='models/', name="anime"),
    "3d":Cartoonizer(dataroot='models/', name="3d"),
    "handdrawn":Cartoonizer(dataroot='models/', name="handdrawn"),
    "sketch":Cartoonizer(dataroot='models/', name="sketch"),
    "artstyle":Cartoonizer(dataroot='models/', name="artstyle"),
    "design":Cartoonizer(dataroot='models/', name="design"),
    "illustration": Cartoonizer(dataroot='models/', name="illustration"),
}
def process():

    for k in list(models.keys()):
        algo = models[k]

        # img = cv2.imread('picture/pic/07.jpg')[...,::-1]

        # result = algo.cartoonize(img)
        result = algo.Convert_video('movie/video/05.mp4')

        # cv2.imwrite(f'picture/{k}.jpg', result[:,:,::-1])
        print('finished!')




if __name__ == '__main__':
    process()
    # concat('picture')



