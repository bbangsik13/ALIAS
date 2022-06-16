'''
    result viewer 
    (c) 2022 March 31 
'''
from flask import Flask, request, jsonify, send_file
from flask import render_template

import json
import os
from PIL import Image
from base64 import encodebytes
import io
from flask_cors import CORS

import sys
import glob

cur_id = 0 
img_dir = None
img_list = []
num_imgs =  len(img_list)
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    global cur_id, img_list, num_imgs
    cur_id = 0
    filename = img_list[cur_id] 
    return render_template('image_viewer.html', imagefile=filename, id=cur_id, total_num = num_imgs)

@app.route('/result/<image_name>', methods=['GET'])
def get_img(image_name):
    image_path = os.path.join(img_dir, image_name)
    return send_file(image_path, mimetype='image/jpg', cache_timeout = 0)  # no cache 

@app.route('/next', methods=['POST'])
def next():
    global cur_id, img_list, num_imgs
    try: 
        step = int(request.form['step'])
        cur_id = (cur_id+step) if (cur_id+step<num_imgs) else (num_imgs-1)
        filename = img_list[cur_id] 
        return render_template('image_viewer.html', imagefile=filename, id=cur_id + 1, total_num = num_imgs)
    except:
        return render_template('image_viewer.html', imagefile=filename, id=0, total_num = 0)


@app.route('/prev', methods=['POST'])
def prev():
    global cur_id, img_list, num_imgs
    try:
        step = int(request.form['step'])
        cur_id = (cur_id-step) if (cur_id-step>=0) else  0
        filename = img_list[cur_id] 
        return render_template('image_viewer.html', imagefile=filename, id=cur_id, total_num = num_imgs)
    except:
        return render_template('image_viewer.html', imagefile=filename, id=0, total_num = 0)

@app.route('/filter', methods=["POST"])
def filter():
    global cur_id, img_list, num_imgs

    try:
        pattern = request.form['pattern']

        img_path = os.path.join(img_dir, pattern + "*.png")
        img_list_f = glob.glob(img_path)  # add png files
        img_path = os.path.join(img_dir, pattern +"*.jpg")
        img_list_f = img_list_f + glob.glob(img_path)   # add jpg files
        if len(img_list_f) > 0: # only when at least one file filtered
            img_list = []  # empty 
            for img_path in img_list_f:
                _, img_name = os.path.split(img_path)
                img_list.append(img_name)
            num_imgs =  len(img_list)
            print("num of images:", len(img_list))
            cur_id = 0
        img_list = sorted(img_list)
        filename = img_list[cur_id] 
        return render_template('image_viewer.html', imagefile=filename, id=cur_id + 1, total_num = num_imgs)
    except:
        return render_template('image_viewer.html', imagefile=filename, id=0, total_num = 0)

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python {} portnum directory".format(sys.argv[0]))
        exit()

    img_dir = sys.argv[2]
    #os.path.join('results_train', 'etri_bot')
    img_path = os.path.join(img_dir, "*.png")
    img_list_f = glob.glob(img_path)  # add png files
    img_path = os.path.join(img_dir, "*.jpg")
    img_list_f = img_list_f + glob.glob(img_path)   # add jpg files
    for img_path in img_list_f:
        _, img_name = os.path.split(img_path)
        img_list.append(img_name)
    img_list = sorted(img_list)
    num_imgs =  len(img_list)
    #print("num of images:", num)

    app.run(host='0.0.0.0', port= int(sys.argv[1]))


