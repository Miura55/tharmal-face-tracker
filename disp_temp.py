# coding: utf-8
import picamera
import picamera.array
import cv2
import pigpio

import Seeed_AMG8833
import os
import time

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import io
from PIL import Image

xsv = 24		#X軸サーボのPort番号
ysv = 25		#y軸サーボのPort番号
span = 300	#サーボのセンターからの可動範囲duty値
xct = 1550	#X軸サーボのセンターduty値
yct = 1549	#X軸サーボのセンターduty値
dly = 0.01  #サーボ駆動時のウェイト時間
stp = 2		#サーボ駆動時のdutyステップ値
xsize = 300	#RGB 水平サイズ
ysize = 300	#RGB 垂直サイズ

#サーボの駆動範囲
xmin = xct - span
xmax = xct + span
ymin = yct - span
ymax = yct + span

#グローバル変数
xpos = xct
ypos = yct
xpos0 = xpos
ypos0 = ypos

sv = pigpio.pi()

#カメラをセンターに移動
sv.set_servo_pulsewidth(xsv, xpos0)
sv.set_servo_pulsewidth(ysv, ypos0)

sensor = Seeed_AMG8833.AMG8833()

# 顔認識モデル
prototxt_path = 'models/deploy.prototxt'
faceDetect_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
faceDetectModel = cv2.dnn.readNetFromCaffe(prototxt_path, faceDetect_path)
faceDetectModel.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# マスクモデル
classes = {0: 'Mask', 1: 'No Mask'}
mask_model_path = 'models/face_mask_detector.onnx'
faceMaskNet = cv2.dnn.readNetFromONNX(mask_model_path)
faceMaskNet.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

def move(svn,in0,in1,step):
	if in1 > in0:
		for duty in range(in0,in1,step):
			sv.set_servo_pulsewidth(svn,duty)
			time.sleep(dly)
	if in1 < in0:
		for duty in range(in0,in1,-step):
			sv.set_servo_pulsewidth(svn,duty)
			time.sleep(dly)


def tharmal_plot(pix):
    temp = np.array([pix[num*8:num*8+8] for num in range(8)])
    temp_array = np.rot90(temp, 2)

    # plot
    plt.figure(figsize=(4, 4), dpi=50)
    sns.heatmap(temp_array, vmax=38, vmin=26, cmap='jet', cbar=False)
    plt.axis("off")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # display
    img_pil = Image.open(buf)
    array_img = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGBA2BGR)
    return array_img, temp_array


def detect_face(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        image, (xsize, ysize)), 1.0, (xsize, ysize), (104.0, 177.0, 123.0), swapRB=True)
    faceDetectModel.setInput(blob)
    detections = faceDetectModel.forward()
    detect_faces = []
    positions = []
    confidences = []
    for i in range(0, detections.shape[2]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        (startX, startY) = (max(0, startX-15), max(0, startY-15))
        (endX, endY) = (min(w-1, endX+15), min(h-1, endY+15))
        confidence = detections[0, 0, i, 2]
        if (confidence > 0.5):
            face = image[startY:endY, startX:endX]
            detect_faces.append(face)
            confidences.append(confidence)
            positions.append((startX, startY, endX, endY))
    if detect_faces:
        max_confidence = max(confidences)
        max_index =  confidences.index(max_confidence)
        return detect_faces[max_index], positions[max_index]
    else:
        return detect_faces, positions


def preprocess(img_data):
    ''' 画像データのスケーリング/正規化 '''
    mean_vec = np.array([0.485, 0.456, 0.406])[::-1]
    stddev_vec = np.array([0.229, 0.224, 0.225])[::-1]
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[2]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[:, :, i] = (
            img_data[:, :, i]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def detect_mask(face):
    face = cv2.resize(face, (224, 224))
    preprocessed = preprocess(face)
    blob = cv2.dnn.blobFromImage(preprocessed)
    faceMaskNet.setInput(blob)
    pred = np.squeeze(faceMaskNet.forward())
    return pred

if __name__ == '__main__':
    with picamera.PiCamera() as camera:
        with picamera.array.PiRGBArray(camera) as stream:
            camera.resolution = (xsize, ysize)
            camera.vflip = True
            camera.hflip = True
            while True:
                # stream.arrayにRGBの順で映像データを格納
                camera.capture(stream, 'bgr', use_video_port=True)
                img = stream.array
                # サーモグラフィーからのヒートマップデータを格納
                pixels = sensor.read_temp()
                tharmal_img, temp_array = tharmal_plot(pixels)

                # 顔の位置と顔画像を取得
                (face, position) = detect_face(img)

                if position:
                    (x, y, w, h) = position
                    prediction = detect_mask(face)
                    max_index = np.argsort(-prediction)[0]
                    label = classes[max_index]
                    try:
                        # 顔の認識範囲に合わせて配列絞り込み->体温を推定
                        face_temp_array = temp_array[int(y/30):int((y+h)/30)][int(x/30):int((x+w)/30)]
                        face_temp = np.amax(face_temp_array)
                        result_text = 'temp: {}'.format(face_temp)
                        print(result_text)
                    except ValueError:
                        pass
                    # 結果をプロット
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    size = 0.7
                    weight = 2
                    color = (157, 216, 100) if label == "Mask" else (71, 99, 255)
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness = weight)

                    #ラベルの作成
                    cv2.rectangle(img, (x, y - 15), (x + 50, y + 5), color, -1)
                    cv2.putText(img, str(face_temp), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # カメラ移動
                    xdf = (x + w/2) - xsize/2
                    ydf = (y + h/2) - ysize/2
                    xpos = int(xpos0 - xdf*0.2)
                    ypos = int(ypos0 + ydf*0.2)
                    if xpos > xmax:
                        xpos = xmax
                    if xpos < xmin:
                        xpos = xmin
                    if ypos > ymax:
                        ypos = ymax
                    if ypos < ymin:
                        ypos = ymin
                    move(xsv,xpos0,xpos,stp)
                    move(ysv,ypos0,ypos,stp)
                    xpos0 = xpos
                    ypos0 = ypos

                # system.arrayをウィンドウに表示
                cv2.imshow('monitor', img)
                cv2.imshow('heatmap', tharmal_img)

                # "q"でウィンドウを閉じる
                if cv2.waitKey(30) == 27:
                    break

                # streamをリセット
                stream.seek(0)
                stream.truncate()
            cv2.destroyAllWindows()
            sv.stop()
