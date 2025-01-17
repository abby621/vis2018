# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import numpy as np
import cv2
import random
import os
import glob
import socket

class CombinatorialTripletSet:
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, num_pos=10, isTraining=True, isOverfitting=False):
        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.isOverfitting = isOverfitting

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.numPos = num_pos
        self.batchSize = batchSize

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line.strip('\n').split(' ')
            # if self.isTraining:
            #     while len(temp) < self.numPos: # make sure we have at least 10 images available per class
            #         temp.append(random.choice(temp))
            if len(temp) > self.numPos:
                self.files.append(temp)
                self.classes.append(ctr)
                ctr += 1

        # if we're overfitting, limit how much data we have per class
        if self.isOverfitting == True:
            self.classes = self.classes[:10]
            self.files = self.files[:10]
            for idx in range(len(self.files)):
                backupFiles = self.files[idx]
                self.files[idx] = backupFiles[:10]

        self.indexes = np.arange(0, len(self.files))

    def getBatch(self):
        numClasses = self.batchSize/self.numPos # need to handle the case where we need more classes than we have?
        classes = np.random.choice(self.classes,numClasses,replace=False)

        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])

        labels = np.zeros([self.batchSize],dtype='int')
        ims = []

        ctr = 0
        for cls in classes:
            random.shuffle(self.files[cls])
            for j in np.arange(self.numPos):
                if j < len(self.files[cls]):
                    img = self.getProcessedImage(self.files[cls][j])
                    if img is not None:
                        batch[ctr,:,:,:] = img
                    labels[ctr] = cls
                    ims.append(self.files[cls][j])
                ctr += 1

        return batch, labels, ims

    def getBatchFromImageList(self,image_list):
        batch = np.zeros([len(image_list), self.crop_size[0], self.crop_size[1], 3])
        for ix in range(0,len(image_list)):
            img = self.getProcessedImage(image_list[ix])
            batch[ix,:,:,:] = img
        return batch

    def getProcessedImage(self, image_file):
        img = cv2.imread(image_file)
        if img is None:
            return None

        if self.isTraining and not self.isOverfitting and random.random() > 0.5:
            img = cv2.flip(img,1)

        # if self.isTraining:
        #     img = doctor_im(img)

        img = cv2.resize(img, (self.image_size[0], self.image_size[1]))

        if self.isTraining and not self.isOverfitting:
            top = np.random.randint(self.image_size[0] - self.crop_size[0])
            left = np.random.randint(self.image_size[1] - self.crop_size[1])
        else:
            top = int(round((self.image_size[0] - self.crop_size[0])/2))
            left = int(round((self.image_size[1] - self.crop_size[1])/2))

        img = img[top:(top+self.crop_size[0]),left:(left+self.crop_size[1]),:]

        return img

class NonTripletSet(CombinatorialTripletSet):
    def __init__(self, image_list, mean_file, image_size, crop_size, batchSize=100, isTraining=True):
        self.image_size = image_size
        self.crop_size = crop_size

        self.meanFile = mean_file
        meanIm = np.load(self.meanFile)

        if meanIm.shape[0] == 3:
            meanIm = np.moveaxis(meanIm, 0, -1)

        self.meanImage = cv2.resize(meanIm, (self.crop_size[0], self.crop_size[1]))

        #img = img - self.meanImage
        if len(self.meanImage.shape) < 3:
            self.meanImage = np.asarray(np.dstack((self.meanImage, self.meanImage, self.meanImage)))

        self.batchSize = batchSize

        self.files = []
        self.classes = []
        # Reads a .txt file containing image paths of image sets where each line contains
        # all images from the same set and the first image is the anchor
        f = open(image_list, 'r')
        ctr = 0
        for line in f:
            temp = line[:-1].split(' ')
            self.files.append(temp)
            self.classes.append(ctr)
            ctr += 1

        self.image_size = image_size
        self.crop_size = crop_size
        self.isTraining = isTraining
        self.indexes = np.arange(0, len(self.files))

    def getBatch(self):
        batch = np.zeros([self.batchSize, self.crop_size[0], self.crop_size[1], 3])
        labels = np.zeros([self.batchSize],dtype='int')
        ims = np.zeros([self.batchSize],dtype=object)

        for ix in range(0,self.batchSize):
            randClass = random.choice(self.classes)
            randIm = random.choice(self.files[randClass])
            randImg = self.getProcessedImage(randIm)
            while randImg is None:
                randClass = np.random.choice(self.classes)
                randIm = random.choice(self.files[randClass])
                randImg = self.getProcessedImage(randIm)

            batch[ix,:,:,:] = randImg
            labels[ix] = randClass
            ims[ix] = randIm

        return batch, labels, ims
