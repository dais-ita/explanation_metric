import numpy as np
import requests
import json
import cv2
import sys
import os
import math
import time
import tensorflow as tf
from keras import backend as K
import pickle
import random
import keras

framework_path = "/home/richard/git/interpretability_framework"
sys.path.append(framework_path)
from DaisFrameworkTool import DaisFrameworkTool

dft = DaisFrameworkTool(explicit_framework_base_path=framework_path)
dataset_json, dataset_tool = dft.LoadFrameworkDataset("CIFAR-10-original")
label_names = [label["label"] for label in dataset_json["labels"]]

source = "train"
train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=False, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source, shuffle=False)
print("num train examples: "+str(len(train_x)))

source = "validation"
val_x, val_y = dataset_tool.GetBatch(batch_size = -1,even_examples=False, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
print("num validation examples: "+str(len(val_x)))

#load train data
source = "test"
test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=False, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
print("num test examples: "+str(len(test_x)))

print("calculating dataset mean")
dataset_tool.dataset_mean = None
dataset_mean = dataset_tool.GetMean()
print(dataset_mean)

print("calculating dataset std")
dataset_tool.dataset_std = None
dataset_std = dataset_tool.GetSTD()
print(dataset_std)

model_name = "vgg16_richard"
model_save_path_suffix = ""

normalise_data = True
model_train_params ={
    "learning_rate": 0.001
    ,"batch_size":64
    ,"num_train_steps":250
    ,"experiment_id":1
}

model_instance = dft.InstantiateModelFromName( \
    model_name, model_save_path_suffix, dataset_json,\
    additional_args = {"learning_rate":model_train_params["learning_rate"]})

model_instance.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=model_instance.learning_rate),
              metrics=['accuracy'])

results = dft.TrainModel(model_instance, \
                         dataset_tool.StandardizeImages(train_x), \
                         train_y, 64, 250, \
                         val_x=dataset_tool.StandardizeImages(val_x), \
                         val_y=val_y, \
                         early_stop=True, save_best_name="vgg16_richard-best")

baseline_accuracy = model_instance.EvaluateModel(\
                    dataset_tool.StandardizeImages(test_x), test_y, model_train_params["batch_size"])

print(baseline_accuracy)