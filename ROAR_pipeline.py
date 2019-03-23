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

from numba import cuda

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
# framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool




#PIXEL LIST FUNCTIONS

def GenerateRandomPixelWeights(images_shape):
    pixel_weight_size = list(images_shape[1:])
    pixel_weight_size[-1] = 1
    
    dataset_pixel_weight_lists = []

    for image_i in range(images_shape[0]):
        if(image_i % 100 == 0):
            print("Generating Random Pixel List for:" + str(image_i))
        pixel_weight_list = CreateOrderedPixelsList(np.random.uniform(size=pixel_weight_size))
        dataset_pixel_weight_lists.append(pixel_weight_list)
        
    return dataset_pixel_weight_lists


def CreateOrderedPixelsList(attribution_map):
    pixel_weight_list = []
    for i in range(attribution_map.shape[0]):
        for j in range(attribution_map.shape[1]):
            #TODO: decide best way to aggregate colour attribution
            if(len(attribution_map[i][j].shape) > 0):
                attribution_value = sum(attribution_map[i][j])
            else:
                attribution_value = attribution_map[i][j]
            pixel_weight_list.append( (i,j,attribution_value) )
    #TODO: confirm not taking the abs is correct
    return sorted(pixel_weight_list,key=lambda x: x[2],reverse=True)
 

def CreatePixelListForAllData(data_x, data_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=None, train_x = None, train_y=None, denormalise_function=None):
    # if creating pixel list for data which isn't the training data, you must also pass the training data so it can be used by some explanation types. 
    # If training x and y not passed then data_x and data_y are the datasets training data.
    if train_x is None:
        train_x = data_x
    
    if train_y is None:
        train_y = data_y
    #default arguments for various explanation techniques 
    if(additional_args is None):
        additional_args = {
        "num_samples":100,
        "num_features":300,
        "min_weight":0.01, 
        "num_background_samples":50,
        "train_x":train_x,
        "train_y":train_y,
        "max_n_influence_images":9,
        "dataset_name":dataset_name,
        "background_image_pool":train_x,
        }
    if(not denormalise_function is None):
        additional_args["denormalise_function"] = denormalise_function
    
    total_imgs = len(data_x)
    dataset_pixel_weight_lists = [] 
    start = time.clock()

    verbose_every_n_steps = 5
    
    reset_session_every = 1 
    #Some explanation implementations cause slow down if they are used repeatidly on the same session.
    #if reset_session_every is trTrueue on the explanation instance, then the session will be cleared and refreshed every 'reset_session_every' steps.

    #TODO: Could be parallelized
    for image_i in range(total_imgs):
        if(image_i % verbose_every_n_steps == 0):
            print(time.clock() - start)

            start = time.clock()
            print("Generating Explanation for Image: "+str(image_i)+ " to "+ str(min(image_i+verbose_every_n_steps, total_imgs))+"/" + str(total_imgs))
            print("")
        
        if(image_i % reset_session_every == 0):
            if(explanation_instance.requires_fresh_session==True):
                if(not framework_tool is None):
                    print("___")
                    print("Resetting Session")
                    model_load_path = model_instance.model_dir
                    del model_instance
                    del explanation_instance
                    tf.reset_default_graph() 
                    tf.keras.backend.clear_session()
                    # print("Releasing GPU")
                    # cuda.select_device(0)
                    # cuda.close()
                    # print("GPU released")
                    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
                    model_instance.LoadModel(model_load_path)
                    explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)
                    print("session restarted")
                    print("___")
                    print("")    
        
        additional_outputs = None
        
        
        image_x = data_x[image_i]
        _, _, _, additional_outputs = explanation_instance.Explain(image_x,additional_args=additional_args) 
        
        attribution_map =  np.array(additional_outputs["attribution_map"])
        pixel_weight_list = CreateOrderedPixelsList(attribution_map)
        
        dataset_pixel_weight_lists.append(pixel_weight_list)


    return dataset_pixel_weight_lists


def SavePixelList(dataset_name,explanation_name,pixel_lists):
    pixel_out_path = dataset_name+"_"+explanation_name+"_"+str(int(time.time()))+".pkl"
    with open(pixel_out_path,"wb") as f:
        pickle.dump(pixel_lists, f)


def LoadPixelListFromPath(pixel_list_path):
    pixel_list = None

    with open(pixel_list_path,"rb") as f:
        pixel_list = pickle.load(f)    
    
    return pixel_list




#LEGACY PERTURBATION FUNCTIONS USING DATASET MEAN
def DeteriateImage(image,image_pixel_list,dataset_mean,deterioration_start_index,deterioration_end):
    # function that specifically deteriorates by the mean
    deterioration_pixels = image_pixel_list[deterioration_start_index:deterioration_end]
    
    for px in deterioration_pixels:
        image[px[0]][px[1]] = [dataset_mean[0],dataset_mean[1],dataset_mean[2]]

    return image


def DeteriateDataset(Xs,deterioration_proportion,dataset_pixel_lists,deterioration_step,deterioration_index_step,dataset_mean):
    deterioration_start_index = int(deterioration_step*deterioration_index_step)
    deterioration_end = int(min(len(dataset_pixel_lists[0])-1, (deterioration_step+1)*deterioration_index_step))
    
    modified_images = []
    
    total_imgs = len(Xs)
    verbose_every_n_steps = 50
    #TODO: Could be parallelized
    for x_i in range(total_imgs):
        if(x_i % verbose_every_n_steps == 0):
            print("Deteriating Image: "+str(x_i)+ " to "+ str(min(x_i+verbose_every_n_steps, total_imgs))+"/" + str(total_imgs))
            print("")
        new_image = DeteriateImage(Xs[x_i], dataset_pixel_lists[x_i],dataset_mean,deterioration_start_index,deterioration_end)
        modified_images.append(new_image)
    
    return np.array(modified_images)




#PERTURBATION FUNCTIONS
def PerturbImage(image,image_pixel_list,perturbation_function,deterioration_start_index,deterioration_end):
    #generic perturbation mangement function that takes a specific perturbation function as an argument
    deterioration_pixels = image_pixel_list[deterioration_start_index:deterioration_end]
    
    for px in deterioration_pixels:
        image = perturbation_function(image, px[0], px[1])
        
    return image


def PerturbData(Xs,deterioration_proportion,dataset_pixel_lists,deterioration_step,deterioration_index_step,perturbation_function):
    deterioration_start_index = int(deterioration_step*deterioration_index_step)
    deterioration_end = int(min(len(dataset_pixel_lists[0]), (deterioration_step+1)*deterioration_index_step))
    
    modified_images = []
    
    total_imgs = len(Xs)
    verbose_every_n_steps = 100
    #TODO: Could be parallelized
    for x_i in range(total_imgs):
        if(x_i % verbose_every_n_steps == 0):
            print("Deteriating Image: "+str(x_i)+ " to "+ str(min(x_i+verbose_every_n_steps, total_imgs))+"/" + str(total_imgs))
            print("")
        new_image = PerturbImage(Xs[x_i], dataset_pixel_lists[x_i],perturbation_function,deterioration_start_index,deterioration_end)
        modified_images.append(new_image)
    
    return np.array(modified_images)


#Perturbation Type Functions
def CreateConstantPeturbFunction(pixel_constant_values):
    def constant_perturbation(image, x, y):
        image[x][y] = [pixel_constant_values[0],pixel_constant_values[1],pixel_constant_values[2]]

        return image
    
    return constant_perturbation




def CreateChildModel(framework_tool, deterioration_proportion, model_train_params):
    model_save_path_suffix = "_"+model_train_params["experiment_id"]+"_"+str(deterioration_proportion)
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
    
    return model_instance




def SaveExperimentResults(output_path,performance_results):
    output_string = ""

    for result in performance_results:
        output_string += str(result[0])+","+str(result[1][0])+","+str(result[1][1])+"\n"
    
    with open(output_path, "w") as f:
        f.write(output_string[:-1])





if __name__ == "__main__":
    #ARGS    
    dataset_name = "Traffic Congestion Image Classification (Resized)"
    dataset_name = "CIFAR-10"
    
    model_name = "vgg16"
    normalise_data = True
    
    explanation_name = "LIME"
    
    experiment_id="testROAR_"+dataset_name
    output_path=str(experiment_id)+"_"+explanation_name+"_results.csv"
    load_base_model_if_exist = True
    save_pixel_list = True

    deterioration_rate = 0.05
    num_deterioration_steps = 20


    generate_random_pixel_list = False
    if(explanation_name == "random"):
        generate_random_pixel_list = True

    load_from_pixel_list_path = "" #make sure to load training set


    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)


    ##DATASET
    dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name)

    label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

   
    #Train
    #TODO: Add optional normalisation
    if(normalise_data):
        denomralise_function = dataset_tool.CreateDestandardizeFuntion()

    #LOAD DATA
    #load all train images as model handles batching
    print("load training data")
    print("")
    source = "train"
    #TODO change batch sizes to -1 , 256 , 256
    train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
    print("num train examples: "+str(len(train_x)))


    #validate on up to 256 images only
    source = "validation"
    val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num test examples: "+str(len(test_x)))
    
    
    print("calculating dataset mean")
    dataset_mean = dataset_tool.GetMean()
    print(dataset_mean)

    if(normalise_data):
        denormalise_function = dataset_tool.CreateDestandardizeFuntion()


    #INSTANTIATE MODEL
    model_train_params ={
    "learning_rate": 0.0001
    ,"batch_size":128
    ,"num_train_steps":150
    ,"experiment_id":experiment_id
    }

    model_save_path_suffix = ""
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
    

    #INSTANTIATE EXPLANTION
    explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)


    #TRAIN OR LOAD MODEL
    model_load_path = model_instance.model_dir
    if(os.path.exists(model_load_path) == True and load_base_model_if_exist == True):
        model_instance.LoadModel(model_load_path)
    else:
        if(normalise_data):
            training_stats = framework_tool.TrainModel(model_instance,dataset_tool.StandardizeImages(train_x), train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= dataset_tool.StandardizeImages(val_x), val_y=val_y)

        else:
            training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)


    #INITAL TEST of MODEL
    test_results = [] # of the form: (proportion_of_deteriated_pixels, test_metrics) . test_metrics = [loss, accuracy] .

    if(normalise_data):
        baseline_accuracy = model_instance.EvaluateModel(dataset_tool.StandardizeImages(test_x), test_y, model_train_params["batch_size"])
    else:
        baseline_accuracy = model_instance.EvaluateModel(test_x, test_y, model_train_params["batch_size"])

    print("metrics", model_instance.model.metrics_names)
    print("baseline_accuracy",baseline_accuracy)

    test_results.append((0,baseline_accuracy))

    #CREATE WORKING COPY OF TRAIN SET
    train_x_deteriated = np.copy(train_x)


    
    num_pixels = dataset_json["image_y"] * dataset_json["image_x"]

    #RESIZE IMAGES IF NEEDED
    #Images may need resizing for model. If that's the case, the framework would have done this automatically during training and explanation generations. 
    #we need to do this manually before deterioration step, the framework model class can do this for us. 
    train_x_deteriated = model_instance.CheckInputArrayAndResize(train_x_deteriated,model_instance.min_height,model_instance.min_width)

    num_pixels_in_padded = train_x_deteriated[0].shape[:-1][0] * train_x_deteriated[0].shape[:-1][1]
    deterioration_index_step = int(math.ceil(num_pixels * deterioration_rate))
    

    
    dataset_pixel_lists = None

    pixel_lists = []
    
    if(generate_random_pixel_list):
        print("Generating Random Pixel List")
        pixel_lists = GenerateRandomPixelWeights(train_x_deteriated.shape)
        if(save_pixel_list):
                print("saving pixel lists")
                SavePixelList("TRAIN_"+dataset_name,explanation_name,pixel_lists)

    else:
        if(load_from_pixel_list_path != ""):
            print("Loading Pixel List")
            pixel_lists = LoadPixelListFromPath(load_from_pixel_list_path)
            print("Pixel List Loaded")
        else:
            print("Creating Pixel List")
            if(normalise_data):
                pixel_lists = CreatePixelListForAllData(dataset_tool.StandardizeImages(train_x_deteriated), test_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=dataset_tool.StandardizeImages(train_x), train_y=train_y, denormalise_function=denormalise_function)
            else:
                pixel_lists = CreatePixelListForAllData(train_x_deteriated, test_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=train_x, train_y=train_y)
                
            if(save_pixel_list):
                print("saving pixel lists")
                SavePixelList("TRAIN_"+dataset_name,explanation_name,pixel_lists)


    #TODO: Could be parallelized
    for deterioration_step in range(num_deterioration_steps):
        print("")
        print("______")
        print("Starting Deteriation Step: "+str(deterioration_step))
        print("")
        #remove the deterioration_index_step * deterioration_step pixels from each image of train
        train_x_deteriated = DeteriateDataset(train_x_deteriated,deterioration_rate,dataset_pixel_lists,deterioration_step,deterioration_index_step,dataset_mean)

        #instantiate child model        
        child_model = CreateChildModel(framework_tool, deterioration_rate*(deterioration_step+1),model_train_params)
        
        #retrain on modified train
        training_stats = framework_tool.TrainModel(child_model,train_x_deteriated, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)

        #retest on test
        new_accuracy = child_model.EvaluateModel(test_x, test_y, model_train_params["batch_size"])
        print("metrics", model_instance.model.metrics_names)
        print("new accuracy",new_accuracy)

        test_results.append( (deterioration_rate*(deterioration_step+1),new_accuracy) )
        
    SaveExperimentResults(output_path,test_results)
