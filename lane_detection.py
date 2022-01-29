import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt
from Lanenet.model2 import Lanenet
from torchvision.transforms import ToTensor, ToPILImage
import time
from utils.evaluation import gray_to_rgb_emb, process_instance_embedding


def region_of_interset(image):
    height = image.shape[0]
    polygon = np.array([
        [(0,256),(110,256),(256,90),(200,90)],
        [(512-110,256),(512,256),(300,90),(256,90)]
        ])
    mask = np.zeros_like(image)
    
    cv2.fillPoly(mask,polygon,(255,255,255))
    # cv2.imshow('mask',mask)
    # print(mask.shape)
    # print(type(mask))
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image


device = torch.device('cuda')
model_path = './TUSIMPLE/Lanenet_output/lanenet_epoch_39_batch_8_AUG.model'
LaneNet_model = Lanenet(2, 4).to(device)
LaneNet_model.load_state_dict(torch.load(model_path, map_location=device))

video_file = '/home/amlab/Chungra_front_detect/701_front_cam/GH020042.MP4'
cap = cv2.VideoCapture(video_file)

img_to_tensor = ToTensor()

if cap.isOpened():
    while True:
        ret, img = cap.read()
        if ret:

            start = time.time()

            x_min, x_max = 650, 970  #height
            y_min, y_max = 600, 1400  #width
            roi_img = img[x_min:x_max, y_min:y_max]

            IMAGE_H = 256
            IMAGE_W = 512
            roi_img = cv2.resize(roi_img, dsize=(IMAGE_W, IMAGE_H), interpolation=cv2.INTER_LINEAR)
            

            # print(type(roi_img))
            # plt.imshow(roi_img)
            # roi_img = cv2.cvtColor(roi_img,cv2.COLOR_BGR2RGB)
            
            dst = np.float32([[200, 80], [260, 80],[0, IMAGE_H] , [IMAGE_W, IMAGE_H]])
            src = np.float32([[45, 0], [512-45, 0], [0, IMAGE_H], [IMAGE_W, IMAGE_H]])
            M = cv2.getPerspectiveTransform(dst, src)
            # Minv = cv2.getPerspectiveTransform(dst, src)

            pre_process_roi_img = roi_img / 127.5 - 1.0
            img_tensor = torch.tensor(pre_process_roi_img,dtype=torch.float)
            img_tensor = np.transpose(img_tensor, (2, 0, 1)).to(device)

            binary_final_logits, instance_embedding = LaneNet_model(img_tensor.unsqueeze(0))
            
            # # gt_image_show = ((img_tensor.numpy() + 1) * 127.5).astype(int)
            # # image_show = gt_image_show.transpose(1,2,0)
            # # image_show = image_show[...,::-1]
            binary_img = torch.argmax(binary_final_logits, dim=1).squeeze().cpu().numpy()

            canvas = np.zeros((256,512,3))
            canvas[np.where(binary_img > 0)] = [255,255,255]

            masked_image = region_of_interset(canvas)
            masked_image = masked_image.astype(np.uint8)
            # print(np.unique(masked_image))
            cv2.imshow('pred',masked_image)
            masked_image_gray = cv2.cvtColor(masked_image,cv2.COLOR_BGR2GRAY)
            # print(masked_image_gray.shape)
            # dst = cv2.addWeighted(roi_img,0.5,masked_image,0.5,0)
            # lines = cv2.HoughLinesP(masked_image_gray, 6, np.pi/180, 150, maxLineGap=10)
            # print(lines)
            # if lines is not None:
            #     for line in lines:
            #         # 검출된 선 그리기 ---③
            #         x1, y1, x2, y2 = line[0]
            #         cv2.line(roi_img, (x1,y1), (x2, y2), (0,255,0), 1)
            cv2.imshow('roi',roi_img)

            # cv2.imshow('pre',dst)

            # print(time.time() - start)
            
            cv2.waitKey(10)
        else:
            pass
            
else:
    print('can not open video')
cap.release()
cv2.destroyAllWindows()
