import glob
import os, csv

train_folder = '/pless_nfs/home/datasets/googleLandmarks/train'
train_file = '/pless_nfs/home/datasets/googleLandmarks/train.csv'

with open(train_file,'rU') as f:
    rd = csv.reader(f,delimiter='\t')
    ims = list(rd)

headers = ims.pop(0)

im_list = []
for im in ims:
    split_im = im[0].replace('"','').split(',')
    new_path = os.path.join(train_folder,split_im[0]+'.jpg')
    if os.path.exists(new_path):
        im_details = (new_path,int(split_im[2]))
        im_list.append(im_details)
        print im_details

ims_by_class = {}
for i in im_list:
    if not i[1] in ims_by_class:
        ims_by_class[i[1]] = []
    ims_by_class[i[1]].append(i[0])

classes = ims_by_class.keys()
numClassesStart = len(classes)
numIms = 0
for cls in classes:
    if len(ims_by_class[cls]) < 10:
        ims_by_class.pop(cls, None)
    else:
        numIms += len(ims_by_class[cls])

classes = ims_by_class.keys()
numClasses = len(classes)

train_path = 'train.txt'
if os.path.exists(train_path):
    os.remove(train_path)

with open(train_path,'a') as train_file:
    for cls in classes:
        im_str = ' '.join(ims_by_class[cls])
        train_file.write(im_str+'\n')
