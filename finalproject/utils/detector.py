# Copyright 2021 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import time
from utils import detect_face
from utils import detect_mask
import tflite_runtime.interpreter as tflite
import platform
import cv2
from threading import Thread
import os
from datetime import datetime
import threading
import requests


# import the necessary packages
#-*-coding:utf8 -*-
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from imutils.video import FPS

import numpy as np
import imutils
import time
from datetime import datetime
import cv2
import os

from gpiozero import Buzzer
from time import sleep
import RPi.GPIO as GPIO
from gpiozero import Servo
import smbus2
class GY906(object):

    MLX90614_RAWIR1=0x04
    MLX90614_RAWIR2=0x05
    MLX90614_TA=0x06
    MLX90614_TOBJ1=0x07
    MLX90614_TOBJ2=0x08
    MLX90614_TOMAX=0x20
    MLX90614_TOMIN=0x21
    MLX90614_PWMCTRL=0x22
    MLX90614_TARANGE=0x23
    MLX90614_EMISS=0x24
    MLX90614_CONFIG=0x25
    MLX90614_ADDR=0x0E
    MLX90614_ID1=0x3C
    MLX90614_ID2=0x3D
    MLX90614_ID3=0x3E
    MLX90614_ID4=0x3F

    def __init__(self, address=0x5a, bus_num=1, units = "c"):
        self.bus_num = bus_num
        self.address = address
        self.bus = smbus2.SMBus(bus=bus_num)
        self.units = units

    def read_reg(self, reg_addr):
        try :
            return self.bus.read_word_data(self.address, reg_addr)
        except:
            return None

    def pass_c(self, celsius):
        return celsius - 273.15

    def pass_k(self, celsius):
        return celsius

    def pass_f(self, celsius):
        return (celsius - 273.15) * 9.0/5.0 + 32

    def data_to_temp(self, data):
        temp = (data*0.02)
        temperature = getattr(self, "pass_" + self.units)(temp)
        return temperature

    def get_amb_temp(self):
        data = self.read_reg(self.MLX90614_TA)
        if data != None:
            return self.data_to_temp(data)
        else:
            return None

    def get_obj_temp(self):
        data = self.read_reg(self.MLX90614_TOBJ1)
        if data != None:
            return self.data_to_temp(data)
        else:
            return None
sensor_IR  = 16
servo = Servo(25)

GPIO.setmode(GPIO.BCM) 
GPIO.setup(sensor_IR ,GPIO.IN)
led_green=18
led_red=19
GPIO.setup(led_green ,GPIO.OUT)
GPIO.setup(led_red ,GPIO.OUT)


units = 'c'

#Bus default = 1
bus = 1
#add another sensor
#bus2 = 3

#address gy906 = 0x5a
address = 0x5a

#GY906
sensor = GY906(address,bus,units)
#add another sensor
#sensor2 = GY906.GY906(address,bus2,units)
buzzer = Buzzer(17)

def create_log(frame, temp, mask, server_url, token):
    status, byte_io = cv2.imencode(".JPEG", frame)
    print("[DEBUG] : convert narray to byte status", status)
    files = {"image": byte_io.tobytes()}
    headers = {"token": token}
    data = {"temp": temp, "mask": mask}
    res = requests.post(server_url + "/hardware/scan-log", files=files, data=data, headers=headers)
    if res.status_code != 200:
      print("[DEBUG] response status not 200 response => ", res.text)
    return res

class Detector():
  """Class for live camera detection"""
  def __init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask, hardware_token, server_url):
    self.cpu_face = cpu_face
    self.cpu_mask = cpu_mask
    self.hardware_token = hardware_token
    self.server_url = server_url

    # path FaceNet
    if self.cpu_face:
      self.MODEL_PATH_FACE = os.path.join(models_path, 'ssd_mobilenet_v2_face_quant_postprocess.tflite')
    else: 
      self.MODEL_PATH_FACE = os.path.join(models_path, 'ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite')
    # path MaskNet
    if self.cpu_mask:
      self.MODEL_PATH_FACE_MASK = os.path.join(models_path, 'mobilenet_v2_mask_classification.tflite')
    else:
      self.MODEL_PATH_FACE_MASK = os.path.join(models_path, 'mobilenet_v2_mask_classification_edgetpu.tflite')

    self.threshold_face = threshold_face
    self.frame_bytes = None
    self.camera = camera
    self.threshold_mask = threshold_mask
    self.mask_labels = ['No Mask', 'Mask']

  def make_interpreter(self, model_file, cpu):
    """Create an interpreter delegating on the tpu or cpu"""
      # set some parameters 
    EDGETPU_SHARED_LIB = {
      'Linux': 'libedgetpu.so.1',
      'Darwin': 'libedgetpu.1.dylib',
      'Windows': 'edgetpu.dll'
    }[platform.system()]

    model_file, *device = model_file.split('@')
    if not cpu:
      interpreter =  tflite.Interpreter(
          model_path=model_file,
          experimental_delegates=[
              tflite.load_delegate(EDGETPU_SHARED_LIB,
                                   {'device': device[0]} if device else {})])
    else:
      interpreter =  tflite.Interpreter(
          model_path=model_file)
    return interpreter

  def thread_function(self):
    GPIO.output(led_green,GPIO.HIGH)
    prev=time.time()
    servo.max()
    while time.time()-prev<5:
          pass
    servo.mid()
    GPIO.output(led_green,GPIO.LOW)
  def thread_function1(self):
    GPIO.output(led_red,GPIO.HIGH)
    prev=time.time()
    buzzer.on()
    while time.time()-prev<5:
          pass
                    #servo.mid()
    GPIO.output(led_red,GPIO.LOW)
    buzzer.off()

  def draw_objects(self, frame, objs, y_mask_pred, fps):
    """Draws the bounding box for each object."""
    for i, obj in enumerate(objs):
        color = (255,255,255)  # white color if mask not classified yet
        bbox = obj.bbox
        # mask detection
        temp = sensor.get_obj_temp()
        
        if temp is not None:
            person_temp = "{0:0.1f}{1}".format(temp,units)
            if len(y_mask_pred) != 0:
              y_pred = y_mask_pred[i]
              label = self.mask_labels[y_pred > self.threshold_mask]

              if label == self.mask_labels[0]:
                color = (0,0,255) # b g r, red color if mask not detected

                cv2.putText(frame, label, (650,650),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "Mask Status : ", (400,650),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "Temp : "+str(person_temp), (400,690),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
              
                cv2.line(frame,(int(1920/6),int(1080/8)),(int((1920/6)+150),int(1080/8)),color,5)
                cv2.line(frame,(int(1920/6),int(1080/8)),(int(1920/6),int(1080/8)+150),color,5)

                cv2.line(frame,(int(1920/6)+450,int(1080/8)),(int(1920/6)+450+150,int(1080/8)),color,5)
                cv2.line(frame,(int(1920/6)+450+150,int(1080/8)),(int(1920/6)+450+150,int(1080/8)+150),color,5)
            
                cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6)+150,int(1080/8)+450),color,5)
                cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6),int(1080/8)+450-150),color,5)

                cv2.line(frame,(int(1920/6)+450,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450),color,5)
                cv2.line(frame,(int(1920/6)+450+150,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450-150),color,5)
                x=Thread(target=self.thread_function1)
                x.start()

              else:
                color = (0,255,0) # b g r, red color if mask not detected

                cv2.putText(frame, label, (650,650),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "Mask Status : ", (400,650),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, "Temp : "+str(person_temp), (400,690),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.line(frame,(int(1920/6),int(1080/8)),(int((1920/6)+150),int(1080/8)),color,5)
                cv2.line(frame,(int(1920/6),int(1080/8)),(int(1920/6),int(1080/8)+150),color,5)

                cv2.line(frame,(int(1920/6)+450,int(1080/8)),(int(1920/6)+450+150,int(1080/8)),color,5)
                cv2.line(frame,(int(1920/6)+450+150,int(1080/8)),(int(1920/6)+450+150,int(1080/8)+150),color,5)
            
                cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6)+150,int(1080/8)+450),color,5)
                cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6),int(1080/8)+450-150),color,5)

                cv2.line(frame,(int(1920/6)+450,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450),color,5)
                cv2.line(frame,(int(1920/6)+450+150,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450-150),color,5)
                x=Thread(target=self.thread_function)
                x.start()   

              # send image to server when mask status is true 
              send_image_thread = SendImageThread(frame.copy(), temp,  "false", self.server_url, self.hardware_token)
              x=Thread(target=send_image_thread.send_image)
              x.start()  

            #cv2.rectangle(frame, (int(bbox[0] - 2), int(bbox[1] - 45)), (int(bbox[2] + 2), int(bbox[1])), color, -1)
          #cv2.putText(frame,
           #       '{} {:.1%}'.format(label, y_pred),
            #      (int(bbox.xmin + 5), int(bbox.ymin - 10)),
             #     cv2.FONT_HERSHEY_SIMPLEX,
              #    2.5 * ((bbox.xmax - bbox.xmin)/frame.shape[0]),
               #   (255,255,255),
                #  2,
                 # cv2.LINE_AA)
          
        
        #cv2.rectangle(frame, 
         #           (int(bbox.xmin), int(bbox.ymin)), 
          #          (int(bbox.xmax), int(bbox.ymax)),
           #        color, 
            #       3)

    cv2.putText(frame, 'FPS:{:.4}'.format(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)


  def start(self):
    """Main loop function."""
    # initialize coral accelerator
    interpreter_face = self.make_interpreter(self.MODEL_PATH_FACE, self.cpu_face)
    interpreter_face.allocate_tensors()

    # initialize face mask classificer
    interpreter_mask = self.make_interpreter(self.MODEL_PATH_FACE_MASK, self.cpu_mask)
    interpreter_mask.allocate_tensors()


    # define some variables
    camera = cv2.VideoCapture(self.camera)
    camera.set(3,1920)
    camera.set(4,1080) 
    cv2.namedWindow('Camera', cv2.WINDOW_FULLSCREEN)

    y_mask_pred = []


    # start loop
    while(True):
      # get opencv data

      
      ret, frame = camera.read()
      cv2.putText(frame, "Mask Status : ", (400,650),cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
      cv2.putText(frame, "Temp : ", (400,690),cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
      cv2.putText(frame, str(datetime.now().replace(microsecond=0)), (920,60),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (54,0,123), 2)
      
      cv2.line(frame,(int(1920/6),int(1080/8)),(int((1920/6)+150),int(1080/8)),(0,150,255),5)
      cv2.line(frame,(int(1920/6),int(1080/8)),(int(1920/6),int(1080/8)+150),(0,150,255),5)

      cv2.line(frame,(int(1920/6)+450,int(1080/8)),(int(1920/6)+450+150,int(1080/8)),(0,150,255),5)
      cv2.line(frame,(int(1920/6)+450+150,int(1080/8)),(int(1920/6)+450+150,int(1080/8)+150),(0,150,255),5)
    
      cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6)+150,int(1080/8)+450),(0,150,255),5)
      cv2.line(frame,(int(1920/6),int(1080/8)+450),(int(1920/6),int(1080/8)+450-150),(0,150,255),5)

      cv2.line(frame,(int(1920/6)+450,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450),(0,150,255),5)
      cv2.line(frame,(int(1920/6)+450+150,int(1080/8)+450),(int(1920/6)+450+150,int(1080/8)+450-150),(0,150,255),5)
      t0 = time.clock()

      frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

      # faces detection
      if GPIO.input(sensor_IR ) !=1:
        print(GPIO.input(sensor_IR ))
        objs = detect_face.predict(interpreter_face, frame_rgb, self.threshold_face)
        # mask detection
        if len(objs) != 0:
          try:
              color=(255,0,0)
              y_mask_pred = detect_mask.predict(interpreter_mask, frame_rgb, objs)

          except:
              y_mask_pred = []
            
          t1 = time.clock()

          self.draw_objects(frame.copy(), objs, y_mask_pred, (1/(t1-t0)))


      cv2.imshow('Camera', frame)



      if cv2.waitKey(1) & 0xFF == ord('q'):
        # When everything done, release the capture
        camera.release()
        cv2.destroyAllWindows()
        break



class Detector_Thread(Thread, Detector):
  """Mutli-Thread class for live camera detection."""
  def __init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask):
    Thread.__init__(self)
    Detector.__init__(self, cpu_face, cpu_mask, models_path, threshold_face, camera, threshold_mask)

  def run(self):
    """Main loop function."""
    # initialize coral accelerator
    interpreter_face = self.make_interpreter(self.MODEL_PATH_FACE, self.cpu_face)
    interpreter_face.allocate_tensors()

    # initialize face mask classificer
    interpreter_mask = self.make_interpreter(self.MODEL_PATH_FACE_MASK, self.cpu_mask)
    interpreter_mask.allocate_tensors()


    # define some variables
    camera = cv2.VideoCapture(self.camera)

    y_mask_pred = []


    # start loop
    while(True):
      # get opencv data
      ret, frame = camera.read()

      t0 = time.clock()

      frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

      # faces detection
      objs = detect_face.predict(interpreter_face, frame_rgb, self.threshold_face)
      cv2.imshow()
      # mask detection
      if len(objs) != 0:
        try:
          y_mask_pred = detect_mask.predict(interpreter_mask, frame_rgb, objs)
          
        except:
          y_mask_pred = []
        
      t1 = time.clock()

      
      self.draw_objects(frame, objs, y_mask_pred, (1/(t1-t0)))
      
      self.frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()

  def get_frame(self):
    """Return a frame for the server"""
    return self.frame_bytes


class SendImageThread():
  def __init__(self, frame, temp, mask, server_url, token):
    self.frame = frame
    self.temp = temp
    self.mask = mask
    self.server_url = server_url
    self.hardware_token = token
  def send_image(self):
    try:
      print('[DEBUG] type of frame', type(self.frame))
      print('[DEBUG] create log variable', self.mask, self.temp, self.server_url, self.hardware_token)
      res = create_log(self.frame, self.temp, self.mask, self.server_url, self.hardware_token)
      print('[DEBUG] create log response', res)
    except Exception as e:
      print('[DEBUG] create log error', e)    