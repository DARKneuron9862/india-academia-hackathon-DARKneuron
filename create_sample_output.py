import tensorflow as tf
import random
import json
# from utils.io import write_json
def read_img(img):
    img=tf.io.read_file(img)
    img=tf.image.decode_jpeg(img,channels=1)
    return img
def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename):
    res = {}
    DARKnet=tf.keras.models.load_model('DARKnet')
    for i in range(1,99):
        image_no = str(i) + '.jpg'
        image=read_img('test/'+image_no)
        result=DARKnet.predict(tf.expand_dims(image/255,axis=0))
        if result>0.5:
            res[image_no]=1
        else:
            res[image_no]=0
    write_json(filename, res)

if __name__ == '__main__':
    
    generate_sample_file('./sample_result1.json')

