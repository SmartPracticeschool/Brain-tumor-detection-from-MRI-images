# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division,print_function
import os
import sys
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from tensorflow.python.keras import backend as K
from tensorflow.python.framework import ops
ops.reset_default_graph()
global graph
#graph=tf.get_default_graph()
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
app=Flask(__name__)
model=load_model("cnn.h5")
print("Model uploaded..Check localhost")
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(64,64))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        
        with graph.as_default():
            preds=model.predict_classes(x)
        index=["Normal","Tumor"]
        text="prediction:"+index[preds[0]]
        return text
        
if __name__=='__main__':
    app.run(debug=False,threaded=False)