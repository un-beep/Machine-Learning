#-*- coding: UTF-8 -*- 


import cv2 as cv2
import os
import tensorflow as tf
import numpy as np
import random


photo_path="./photo"
dirs={"1","2","3","4","5"}




for inedex,name in enumerate(dirs):
    with tf.Session() as sess:
        class_path=photo_path+'/'+name
        dir_path=class_path+"new"
        image_list=[]
        print(dir_path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for img_name in os.listdir(class_path):
            filenames=class_path+'/'+img_name
            print(filenames)
            img=cv2.imread(filenames)
            size=img.shape
            h=size[0]
            w=size[1]
            #cv2.imshow("a",img)      
            #cv2.waitKey(0)
            if h>w:
                crop_img=img[int(h/2-w/2):int(h/2+w/2),0:w]
                scale_img=cv2.resize(crop_img,(299,299))
                #cv2.imshow("a",scale_img)      
                #cv2.waitKey(0)
                cv2.imwrite(dir_path+"/new"+img_name+".jpg",scale_img)
                
                
                dimg=tf.image.flip_up_down(scale_img)
                dimg=dimg.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"0"+".jpg",dimg)

                dimg2=tf.image.flip_left_right(scale_img)
                dimg2=dimg2.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"1"+".jpg",dimg2)

                
                dimg3=tf.image.transpose_image(scale_img)
                dimg3=dimg3.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"2"+".jpg",dimg3)


                dimg4=tf.image.random_brightness(scale_img,0.2)
                dimg4=tf.image.random_saturation(dimg4,1,5)
                dimg4=tf.image.random_hue(dimg4,0.1)
                dimg4=dimg4.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"3"+".jpg",dimg4)
            
                        
            else:
                crop_img=img[0:h,int(w/2-h/2):int(w/2+h/2)]
                #print(crop_img.shape)
                scale_img=cv2.resize(crop_img,(299,299))
                #cv2.imshow("a",scale_img)      
                #cv2.waitKey(0)
                cv2.imwrite(dir_path+"/new"+img_name+".jpg",scale_img)

                dimg=tf.image.flip_up_down(scale_img)
                dimg=dimg.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"0"+".jpg",dimg)

                dimg2=tf.image.flip_left_right(scale_img)
                dimg2=dimg2.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"1"+".jpg",dimg2)

                
                dimg3=tf.image.transpose_image(scale_img)
                dimg3=dimg3.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"2"+".jpg",dimg3)


                dimg4=tf.image.random_brightness(scale_img,0.2)
                dimg4=tf.image.random_saturation(dimg4,1,5)
                dimg4=tf.image.random_hue(dimg4,0.1)
                dimg4=dimg4.eval()
                cv2.imwrite(dir_path+"/new"+img_name+"3"+".jpg",dimg4)

