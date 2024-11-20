#coding:utf-8

import numpy as np
import tornado.web
import tornado.ioloop
import tornado.httpserver
import tornado.options
import tornado.websocket
from tornado.options import options, define
from tornado.web import RequestHandler, MissingArgumentError
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import json
import cv2
from tensorflow.python.platform import gfile
import json

labels=["紫娇花","石榴花","酢浆草","四季海棠","万寿菊"]


# 下载的谷歌训练好的inception-v3模型文件名
MODEL_FILE = './inceptionV3/tensorflow_inception_graph.pb'
# inception-v3 模型中代表瓶颈层结果的张量名称
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
# 图像输入张量所对应的名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
# inception-v3 模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

define("port", default=8000, type=int, help="run server on the given port.")

imgnum=0

def create_inception_graph():
    #加载已训练好的inception-v3模型
    with tf.Graph().as_default() as graph:
        with gfile.FastGFile(MODEL_FILE, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    return graph, bottleneck_tensor, jpeg_data_tensor



def recflower(thedir):
    img_raw_data=gfile.FastGFile(thedir, 'rb').read()
    graph, bottleneck_tensor, jpeg_data_tensor =create_inception_graph()
    with tf.Session(graph=graph) as sess:
        init = tf.global_variables_initializer()
        print("inception-v3")
        sess.run(init)
        image_value = sess.run(bottleneck_tensor,{jpeg_data_tensor:img_raw_data})
        image_value = np.squeeze(image_value)
        image_value= np.reshape(image_value,[1,2048])
        print(image_value)

    #bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
    #ground_truth_input = tf.placeholder(tf.int64, [None], name='GroundTruthInput')
    
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = './train_dir/model.pb'
    

    with open(output_graph_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        input1=sess.graph.get_tensor_by_name("BottleneckInputPlaceholder:0")
        output=sess.graph.get_tensor_by_name("output/prob:0")
        result=sess.run(output,{input1:image_value})
        print(result)
    flowername=labels[np.argmax(result)]
    print(flowername)
    return flowername

    
class IndexHandler(RequestHandler):    
    def get(self):
        self.render('index.html',title='输入数字')

class UploadHandler(RequestHandler):    
    def post(self):
        global imgnum
        print("receive image")
        b64str=self.request.body.decode()
        b64str=json.loads(b64str)
        image=b64str.get("d")
        imgdata=image.split(";base64,")[1]
        imgdata=base64.b64decode(imgdata)
        f=open("./receive/"+str(imgnum)+".jpg","wb")
        f.write(imgdata)
        pic=cv2.imread("./receive/"+str(imgnum)+".jpg")
        '''cv2.imshow("rec",pic)
        cv2.waitKey(0)'''
        str1=recflower("./receive/"+str(imgnum)+".jpg")
        imgnum=imgnum+1
        self.write(str1)

if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/uploadImg", UploadHandler),
    ])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.current().start()
