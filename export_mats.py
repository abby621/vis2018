# python export_mats.py traffickcam 3 True
import tensorflow as tf
from classfile import NonTripletSet
import os, random, time
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow.contrib.slim as slim
from nets import resnet_v2
from scipy.io import savemat
import pickle
import sys
import glob
from shutil import copyfile

def main(dataset,whichGPU,is_finetuning):
    print dataset, whichGPU, is_finetuning
    if is_finetuning.lower() == 'true':
        nets_dir = os.path.join('./output',dataset,'ckpts','finetuning')
        outMatFolder = os.path.join('./output','mats',dataset,'finetuning')
    else:
        nets_dir = os.path.join('./output',dataset,'ckpts','fromScratch')
        outMatFolder = os.path.join('./output','mats',dataset,'finetuning')

    test_file = os.path.join('./input',dataset,'test.txt')
    mean_file = os.path.join('./input',dataset,'meanIm.npy')

    if not os.path.exists(outMatFolder):
        os.makedirs(outMatFolder)

    img_size = [256, 256]
    crop_size = [224, 224]
    batch_size = 100
    output_size = 128

    # Create test_data "batcher"
    #train_data = CombinatorialTripletSet(train_file, mean_file, img_size, crop_size, batch_size, num_pos_examples,isTraining=False)
    test_data = NonTripletSet(test_file, mean_file, img_size, crop_size, batch_size,isTraining=False)

    image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
    repMeanIm = np.tile(np.expand_dims(test_data.meanImage,0),[batch_size,1,1,1])
    final_batch = tf.subtract(image_batch,repMeanIm)
    label_batch = tf.placeholder(tf.int32, shape=(batch_size))

    print("Preparing network...")
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=False)

    featLayer = 'resnet_v2_50/logits'
    non_norm_feat = tf.squeeze(layers[featLayer])
    feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
    convOut = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0"))
    weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/weights:0"))
    biases = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/biases:0"))
    gap = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"))

    ims_and_labels_path = os.path.join('./output',dataset,'ims_and_labels.pkl')
    if not os.path.exists(ims_and_labels_path):
        testingImsAndLabels = [(test_data.files[ix][iy],test_data.classes[ix]) for ix in range(len(test_data.files)) for iy in range(min(10,len(test_data.files[ix])))]
        numTestingIms = batch_size*(len(testingImsAndLabels)/batch_size)
        testingImsAndLabels = testingImsAndLabels[:numTestingIms]
        with open(ims_and_labels_path, 'wb') as fp:
            pickle.dump(testingImsAndLabels, fp)
    else:
        with open (ims_and_labels_path, 'rb') as fp:
            testingImsAndLabels = pickle.load(fp)
        numTestingIms = len(testingImsAndLabels)

    print 'Num Images: ',numTestingIms

    c = tf.ConfigProto()
    c.gpu_options.visible_device_list=whichGPU

    sess = tf.Session(config=c)
    saver = tf.train.Saver(max_to_keep=100)

    snapshot_base = "-".join(max(glob.glob(os.path.join(nets_dir,'checkpoint-*.index')),key=os.path.getctime).split('-')[:2])
    print 'Snapshot base: ',snapshot_base
    all_snapshots = glob.glob(snapshot_base+'*.index')
    print 'All snapshots: ',all_snapshots

    snapshot_iters = [4999,9999,14999,19999,24999,49999,74999,99999]

    pretrained_nets = []
    for snapshot in all_snapshots:
        numIters = int(snapshot.split('-')[-1].split('.index')[0])
        if numIters in snapshot_iters:
            iterFolder = os.path.join(outMatFolder,str(numIters))
            if not os.path.exists(iterFolder):
                os.makedirs(iterFolder)
            if len(os.listdir(iterFolder)) != len(test_data.files):
                pretrained_nets.append(snapshot.split('.index')[0])

    if is_finetuning:
        iterFolder = os.path.join(outMatFolder,str(0))
        if not os.path.exists(iterFolder):
            os.makedirs(iterFolder)
        if len(os.listdir(iterFolder)) != len(test_data.files):
            pretrained_nets.append('./models/ilsvrc.ckpt')

    outImDir = os.path.join('./output','images',dataset)
    if not os.path.exists(outImDir):
        os.makedirs(outImDir)

    print 'Pretrained nets: ',pretrained_nets
    for pretrained_net in pretrained_nets:
        if not 'ilsvrc' in pretrained_net:
            numIters = int(pretrained_net.split('-')[-1].split('.index')[0])
        else:
            numIters  = 0
            output_size = 1001
            tf.reset_default_graph()
            image_batch = tf.placeholder(tf.float32, shape=[batch_size, crop_size[0], crop_size[0], 3])
            repMeanIm = np.tile(np.expand_dims(test_data.meanImage,0),[batch_size,1,1,1])
            final_batch = tf.subtract(image_batch,repMeanIm)
            label_batch = tf.placeholder(tf.int32, shape=(batch_size))
            with slim.arg_scope(resnet_v2.resnet_arg_scope()):
                _, layers = resnet_v2.resnet_v2_50(final_batch, num_classes=output_size, is_training=False)

            featLayer = 'resnet_v2_50/logits'
            non_norm_feat = tf.squeeze(layers[featLayer])
            feat = tf.squeeze(tf.nn.l2_normalize(layers[featLayer],3))
            convOut = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/postnorm/Relu:0"))
            weights = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/weights:0"))
            biases = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/logits/biases:0"))
            gap = tf.squeeze(tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0"))

            sess = tf.Session(config=c)
            saver = tf.train.Saver(max_to_keep=100)

        saver.restore(sess, pretrained_net)

        testingFeats = np.empty((numTestingIms,feat.shape[1]),dtype=np.float32)
        testingCV = np.empty((numTestingIms,convOut.shape[1]*convOut.shape[2],convOut.shape[3]),dtype=np.float32)
        testingGAP = np.empty((numTestingIms,gap.shape[1]),dtype=np.float32)
        testingIms = np.empty((numTestingIms),dtype=object)
        testingLabels = np.empty((numTestingIms),dtype=np.int32)
        for idx in range(0,numTestingIms,batch_size):
            print idx, '/', numTestingIms
            il = testingImsAndLabels[idx:idx+batch_size]
            ims = [i[0] for i in il]
            labels = [i[1] for i in il]
            batch = test_data.getBatchFromImageList(ims)
            testingLabels[idx:idx+batch_size] = labels
            new_ims = []
            for im,cls in zip(ims,labels):
                imdir = os.path.join(outImDir,str(cls))
                new_path=os.path.join(imdir,im.split('/')[-1])
                new_ims.append(new_path)

            testingIms[idx:idx+batch_size] = new_ims
            ff, gg, cvOut, wgts, bs = sess.run([non_norm_feat,gap,convOut,weights,biases], feed_dict={image_batch: batch, label_batch:labels})
            testingFeats[idx:idx+batch_size,:] = np.squeeze(ff)
            testingGAP[idx:idx+batch_size,:] = np.squeeze(gg)
            testingCV[idx:idx+batch_size,:,:] = cvOut.reshape((cvOut.shape[0],cvOut.shape[1]*cvOut.shape[2],cvOut.shape[3]))

        for cls in np.unique(testingLabels):
            inds = np.where(testingLabels==cls)[0]
            out_data = {}
            out_data['ims'] = testingIms[inds]
            out_data['labels'] = testingLabels[inds]
            out_data['feats'] = testingFeats[inds,:]
            out_data['gap'] = testingGAP[inds,:]
            out_data['conv'] = testingCV[inds,:,:]
            out_data['weights'] = wgts
            out_data['biases'] = bs

            iterFolder = os.path.join(outMatFolder,str(numIters))
            outfile = os.path.join(iterFolder,str(cls)+'.mat')
            savemat(outfile,out_data)
            print outfile

    for im in testingImsAndLabels:
        cls = im[1]
        imdir = os.path.join(outImDir,str(cls))
        if not os.path.exists(imdir):
            os.makedirs(imdir)
        new_path=os.path.join(imdir,im[0].split('/')[-1])
        if not os.path.exists(new_path):
            copyfile(im[0],new_path)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 4:
        print 'Expected input parameters: dataset,whichGPU,is_finetuning'
    dataset = args[1]
    whichGPU = args[2]
    is_finetuning = args[3]
    main(dataset,whichGPU,is_finetuning)

# rsync -avz astylianou@focus.cse.wustl.edu:/project/focus/abby/tc_feats/pretrained/ /Users/abby/Documents/tc_feats/pretrained_mats
# rsync -avz astylianou@focus.cse.wustl.edu:/project/focus/abby/tc_feats/images/ .
# import glob
# import shutil
# import os
#

# For traffickcam, split into traffickcam and expedia images
# ims = glob.glob('./*/*.jpg')
# for im in ims:
#     split_im = im.split('/')
#     cls = split_im[1]
#     tc_path = os.path.join(split_im[0],split_im[1],'traffickcam')
#     ex_path = os.path.join(split_im[0],split_im[1],'expedia')
#     if not os.path.exists(tc_path):
#         os.makedirs(tc_path)
#     if not os.path.exists(ex_path):
#         os.makedirs(ex_path)
#     if len(split_im[2]) > 12:
#         new_path = os.path.join(tc_path,split_im[2])
#     else:
#         new_path = os.path.join(ex_path,split_im[2])
#     shutil.move(im,new_path)
