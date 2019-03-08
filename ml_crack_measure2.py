from crackmeasure import Line
from test_rekt import rect_select
import matplotlib.pyplot as plt
from scipy import signal as sg
import numpy as np
import cv2
import json
import os
import glob

dirr = "/home/nolan/Desktop/101018_HNS_2-4/190N/"
dir_template = "/home/nolan/Desktop/101018_HNS_2-4/%s/panorama.tif"
dir_template2 = "/home/nolan/Desktop/10318_1-2/%s/panorama.tif"

im = plt.imread("/home/nolan/Desktop/101018_HNS_2-4/190N/panorama.tif")

dirs = ['190N', '170N', '139N', '118N', '99N', '78N', '61N', '41N', '0N']

dirs2 = ["180N","160N","140N","120N","100N","80N","60N"]


Ly = np.array([[ 1,  2,  1],
               [ 0,  0,  0],
               [-1, -2, -1]])

Lx = np.array([[ 1,  0, -1],
               [ 2,  0, -2],
               [ 1,  0, -1]])

#frame = im[1000:4000]

#Gx = sg.convolve(frame,Lx)
#Gy = sg.convolve(frame,Ly)
#G = np.sqrt(Gx**2 + Gy**2)
#del im

def line_from_json(class_dict):
    return Line(class_dict['x1'],
                class_dict['y1'],
                class_dict['x2'],
                class_dict['y2'])
        

def linelist_from_json(fname):
    with open(fname,'r') as fp:
        data = json.load(fp)
        return [line_from_json(l) for l in data]

def get_training_points(directory):
    points = []
    files = glob.glob(os.path.join(directory,
                                "panorama.tif.crack*.json"))
    for f in files:
        data = json.load(open(f))
        for d in data:
            y1 = d['y1']
            y2 = d['y2']
            x1 = d['x1']
            x2 = d['x2']
            if y1 > y2:
                tmp = x1
                x1 = x2
                x2 = tmp
                tmp = y1
                y1 = y2
                y2 = tmp
            points.append([x2,y2])
    return points

def get_frame(point,points,im,extent=(100,100)):
    contains = []
    xc = int(point[0])
    yc = int(point[1])
    x1 = xc - extent[0]//2
    x2 = xc + extent[0]//2
    y1 = yc - extent[1]//2
    y2 = yc + extent[1]//2
    for p in points:
        if p[0] > x1 and p[0] < x2 and p[1] > y1 and p[1] < y2:
            p_new = [p[0] - xc + extent[0]//2, p[1] - yc+extent[1]//2]
            contains.append(p_new)
    frame = im[y1:y2][:,x1:x2]
    return frame, contains


def convolutions(arr):
    Gx = sg.convolve(arr,Lx,mode='same')
    Gy = sg.convolve(arr,Ly,mode='same')
    G = np.sqrt(Gx**2 + Gy**2)
    return Gx,Gy,G

def convolutions2(arr):
    blur = blur = cv2.GaussianBlur(frame,(5,5),0)
    Gy = cv2.Sobel(blur,cv2.CV_64F,0,1)
    Gx = cv2.Sobel(blur,cv2.CV_64F,1,0)
    G = np.sqrt(Gx**2 + Gy**2)
    return blur,Gy,Gx,G

def convolutions_norm(arr):
    blur = cv2.GaussianBlur(arr,(5,5),0)
    Gy = cv2.Sobel(blur,cv2.CV_64F,0,1)
    Gx = cv2.Sobel(blur,cv2.CV_64F,1,0)
    G = np.sqrt(Gx**2 + Gy**2)
    median = np.median(arr)
    return blur/median,Gx/median,Gy/median,G/median

def crack_search(arr,darr,window_size,min_shade,max_diff):
    points = []
    points2 = []
    for i in range(0,len(arr)-window_size,window_size):
        window = arr[i:i+window_size]
        dwindow = darr[i:i+window_size]
        if np.amin(window) < min_shade and np.amax(dwindow) > max_diff and np.abs(np.amin(dwindow)) > max_diff:
            if np.argmin(dwindow) < np.argmax(dwindow):  
                points.append(i+np.argmax(dwindow))
                points2.append(i+np.argmin(dwindow))
    return points,points2

def look_for_cracks(arr,y1,y2,window_size=100,min_shade=0.7,
                    max_diff=0.5,step =10):
    blur,Gx,Gy,G = convolutions_norm(arr)
    dys = []
    for i in range(0,arr.shape[1],10):
        cracks,cracks2 = crack_search(blur[:,i],Gy[:,i],
                                      window_size,min_shade,max_diff)
        for j in range(len(cracks)):
            l = Line(i,cracks[j],i,cracks2[j])
            dys.append(l)
	#_=plt.plot(i*np.ones(len(cracks)),cracks,'ro')
	#_ = plt.plot(i*np.ones(len(cracks2)),cracks2,'bo')
    return dys

def search_frame(im):
    for i in range(1000,im.shape[0]-3000,3000):
        frame = im[i:i+3000]
        dys = look_for_cracks(frame,750,1000,min_shade=0.5,max_diff=0.45)
        plt.imshow(frame)
        for l in dys:
            _ = plt.plot([l.x1],[l.y1],'ro')
            _ = plt.plot([l.x2],[l.y2],'bo')
        plt.show()


'''
for i in range(1000,im.shape[0]-3000,3000):
	frame = im[i:i+3000]
	dys = look_for_cracks(frame,750,1000,min_shade=0.5,max_diff=0.45)
	plt.imshow(frame)
	for l in dys:
		_ = plt.plot([l.x1],[l.y1],'ro')
		_ = plt.plot([l.x2],[l.y2],'bo')
	plt.show()
'''

def test(fam):
    b = look_for_cracks(fam,750,1000,min_shade=0.5,max_diff=0.45)
    plt.imshow(fam)
    for l in b:
        _ = plt.plot([l.x1],[l.y1],'ro')
        _ = plt.plot([l.x2],[l.y2],'bo')
    plt.show()

def test2(imm):
    for i in range(1000,imm.shape[0]-3000,3000):
        fram = imm[i:i+3000]
        test(fram)

def get_cracks(imm,min_shade=0.5,max_diff=0.45):
    points = []
    for i in range(0,imm.shape[0]-3000,3000):
        fram = imm[i:i+3000]
        b = look_for_cracks(fram,0,fram.shape[1],min_shade=min_shade,max_diff=max_diff)
        for l in b:
            l.add_offset(0,i)
        points.extend(b)
    return points
    

def json_dump(iterable, fname):
    with open(fname,'w') as fp:
        json.dump([i.__dict__ for i in iterable],fp)


def run():
    fig,ax = plt.subplots()  
    for d in dirs2:
        fn = dir_template2%(d)
        jsfn=fn.replace(".tif",".mlcracks.json")
        pts = linelist_from_json(jsfn)
        x = []
        y = []
        for p in pts:
            x.append(p.x1)
            x.append(p.x2)
            y.append(p.y1)
            y.append(p.y2)
        ax.plot(x,y,'o')
    return rect_select(ax)
    
def remove_outliers(arr):
    iqr = np.percentile(arr,75) - np.percentile(arr,25)
    med = np.median(arr)
    for i,el in enumerate(arr):
        if np.absolute(el - med) > (1.5*iqr):
            arr.pop(i)
            
