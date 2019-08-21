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

import random


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

###UPDATE FRAMEWORK PATH
framework_path = "/home/richard/git/interpretability_framework"
# framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool


np.random.seed(42)
tf.set_random_seed(1234)
random.seed(1234)


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


def CreateOrderedPixelsList(attribution_map, abs=False):
    pixel_weight_list = []
    for i in range(attribution_map.shape[0]):
        for j in range(attribution_map.shape[1]):
            #TODO: decide best way to aggregate colour attribution
            if(len(attribution_map[i][j].shape) > 0):
                attribution_value = sum(attribution_map[i][j])
            else:
                attribution_value = attribution_map[i][j]
            if abs:
                attribution_value = np.abs(attribution_value)
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
        "num_background_samples":500,
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
    
    reset_session_every = 5
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
    pixel_out_path = os.path.join("pixel_lists",dataset_name+"_"+explanation_name+"_abs_"+str(int(time.time()))+".pkl")
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


def PerturbData(Xs,deterioration_proportion,dataset_pixel_lists,deterioration_step,deterioration_index_step,perturbation_function, deletion_game=True, total_num_deterioration_steps=20):
    if(deletion_game):
        print("Using deletion game metric")
        deterioration_start_index = int(deterioration_step*deterioration_index_step)
        deterioration_end = int(min(len(dataset_pixel_lists[0]), (deterioration_step+1)*deterioration_index_step))
    else:
        print("Using preservation game metric")
        deterioration_start_index = int((total_num_deterioration_steps - (deterioration_step+1)) * deterioration_index_step)
        deterioration_end = int(min(len(dataset_pixel_lists[0]), (total_num_deterioration_steps - deterioration_step) * deterioration_index_step))


    modified_images = []
    
    total_imgs = len(Xs)
    verbose_every_n_steps = 500
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
    def constant_perturbation(img, x, y):
        img[x][y] = [pixel_constant_values[0],pixel_constant_values[1],pixel_constant_values[2]]

        return img
    
    return constant_perturbation

def DeteriorateImageWithRandomColour(img,x,y):
    img[x][y] = [random.random(),random.random(),random.random()]
    
    return img


def CreateGridPerturbationFunction(grid_width=3,grid_height=3, pixel_operation_function=DeteriorateImageWithRandomColour):
    grid_width_distance = int((grid_width-1) / 2)
    grid_height_distance = int((grid_height-1) / 2)
    print(grid_height_distance)
    def DeteriorateGridOfImageWithRandomColour(img,x,y):
        for width_modifier in range(-grid_width_distance,(grid_width_distance+1),1):
            for height_modifier in range(-grid_height_distance,(grid_height_distance+1),1):
                img[x+width_modifier][y+height_modifier] = [random.random(),random.random(),random.random()]
                # img = pixel_operation_function(img, x+width_modifier,y+height_modifier)
                
        return img
    

    return DeteriorateImageWithRandomColour



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
    
    explanation_names = ["random","Shap"] #"LIME","Shap"
    load_from_pixel_list_path_dict={
        "LIME": os.path.join("pixel_lists","TRAIN_testROAR_CIFAR-10_LIME_1553461654.pkl")
        ,"Shap": os.path.join("pixel_lists","TRAIN_testROAR_CIFAR-10_Shap_1553686507.pkl")
        ,"random": os.path.join("pixel_lists","TRAIN_testROAR_CIFAR-10_random_1553734921.pkl")
    }

    experiment_id="testROAR_PRESERVATION_"+dataset_name
    load_base_model_if_exist = True
    save_pixel_list = True

    use_deletion_game = False
    deterioration_rate = 0.05
    num_deterioration_steps = 20

    #SINGLE EXPLANATION VARIABLES
    # explanation_name = "LIME"
    
    # output_path=str(experiment_id)+"_"+explanation_name+"_results.csv"
    # generate_random_pixel_list = False
    # if(explanation_name == "random"):
    #     generate_random_pixel_list = True

    # load_from_pixel_list_path = "" #make sure to load a TRAINing set


    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)


    ##DATASET
    dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name)

    label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

   
    #Train
    
    #LOAD DATA
    #load all train images as model handles batching
    print("load training data")
    print("")
    source = "train"
    #TODO change batch sizes to -1 , 256 , 256
    train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source, shuffle=False)
    print("num train examples: "+str(len(train_x)))


    #validate on up to 256 images only
    source = "validation"
    val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
    print("num test examples: "+str(len(test_x)))
    
    print("First and Last Images in Data:")
    print("Train:")
    print(dataset_tool.live_training[0])
    print(dataset_tool.live_training[-1])
    print("")
    print("Test:")
    print(dataset_tool.live_test[0])
    print(dataset_tool.live_test[-1])
    print("")
    

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
    
    
    #FOR EACH EXPLANATION
    for explanation_name in explanation_names:
    
        model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = model_train_params)
        
        #TRAIN OR LOAD MODEL
        model_load_path = model_instance.model_dir
        if(os.path.exists(model_load_path) == True and load_base_model_if_exist == True):
            model_instance.LoadModel(model_load_path)
        else:
            if(normalise_data):
                training_stats = framework_tool.TrainModel(model_instance,dataset_tool.StandardizeImages(train_x), train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= dataset_tool.StandardizeImages(val_x), val_y=val_y)

            else:
                training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)

        print("Generating Results for: " + explanation_name)
        load_from_pixel_list_path = ""
        
        if(explanation_name in load_from_pixel_list_path_dict):
            load_from_pixel_list_path = load_from_pixel_list_path_dict[explanation_name]

        output_path=os.path.join("results",str(experiment_id)+"_"+explanation_name+"_results.csv")
        
        generate_random_pixel_list = False
        if(explanation_name == "random" and load_from_pixel_list_path == ""):
            generate_random_pixel_list = True
        
        #INSTANTIATE EXPLANTION
        if(explanation_name != "random"):
            explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)


        #INITAL TEST of MODEL
        test_results = [] # of the form: (proportion_of_deteriated_pixels, test_metrics) . test_metrics = [loss, accuracy] .

        if(normalise_data):
            baseline_accuracy = model_instance.EvaluateModel(dataset_tool.StandardizeImages(test_x), test_y, model_train_params["batch_size"])
        else:
            baseline_accuracy = model_instance.EvaluateModel(test_x, test_y, model_train_params["batch_size"])

        print("metrics", model_instance.model.metrics_names)
        print("baseline_accuracy",baseline_accuracy)

        test_results.append((0,baseline_accuracy))
        
        #TODO: record image wise predictions for baseline
       
        #CREATE WORKING COPY OF TRAIN SET
        x_deteriated = np.copy(train_x)


        
        num_pixels = dataset_json["image_y"] * dataset_json["image_x"]

        #RESIZE IMAGES IF NEEDED
        #Images may need resizing for model. If that's the case, the framework would have done this automatically during training and explanation generations. 
        #we need to do this manually before deterioration step, the framework model class can do this for us. 
        x_deteriated = model_instance.CheckInputArrayAndResize(x_deteriated,model_instance.min_height,model_instance.min_width)

        num_pixels_in_padded = x_deteriated[0].shape[:-1][0] * x_deteriated[0].shape[:-1][1]
        deterioration_index_step = int(math.ceil(num_pixels * deterioration_rate))
        

        
        
        pixel_lists = []
        
        if(generate_random_pixel_list):
            print("Generating Random Pixel List")
            pixel_lists = GenerateRandomPixelWeights(x_deteriated.shape)
            if(save_pixel_list):
                    print("saving pixel lists")
                    SavePixelList("TRAIN_"+experiment_id,explanation_name,pixel_lists)

        else:
            if(load_from_pixel_list_path != ""):
                print("Loading Pixel List")
                pixel_lists = LoadPixelListFromPath(load_from_pixel_list_path)
                print("Pixel List Loaded")
            else:
                print("Creating Pixel List")
                if(normalise_data):
                    pixel_lists = CreatePixelListForAllData(dataset_tool.StandardizeImages(x_deteriated), test_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=dataset_tool.StandardizeImages(train_x), train_y=train_y, denormalise_function=denormalise_function)
                else:
                    pixel_lists = CreatePixelListForAllData(x_deteriated, test_y, dataset_name, model_instance, explanation_instance,additional_args=None,framework_tool=framework_tool, train_x=train_x, train_y=train_y)
                    
                if(save_pixel_list):
                    print("saving pixel lists")
                    SavePixelList("TRAIN_"+experiment_id,explanation_name,pixel_lists)


        perturbation_function = CreateConstantPeturbFunction(dataset_mean)


        #TODO: Could be parallelized
        #TODO: Do we need to recreate pixel list on detriorated images (artifacting / new use of redundant features)
        for deterioration_step in range(num_deterioration_steps):
            deterioration_output_path = str(experiment_id)+"_"+explanation_name+"_"+format(deterioration_step, '05d')+"_deterioration_results.csv"
            deterioration_output_path = os.path.join("results","image_results",deterioration_output_path)
                
            deterioration_output_string = ""
            print("")
            print("______")
            print("Starting deterioration Step: "+str(deterioration_step))
            print("")
            
            #remove the deterioration_index_step * deterioration_step pixels from each image of train
            x_deteriated = PerturbData(x_deteriated,deterioration_rate,pixel_lists,deterioration_step,deterioration_index_step,perturbation_function, deletion_game=use_deletion_game, total_num_deterioration_steps=num_deterioration_steps)
                
            # x_deteriated = DeteriateDataset(x_deteriated,deterioration_rate,pixel_lists,deterioration_step,deterioration_index_step,dataset_mean)

            #instantiate child model        
            child_model = CreateChildModel(framework_tool, deterioration_rate*(deterioration_step+1),model_train_params)
            
            #retrain on modified train
            if(normalise_data):
                training_stats = framework_tool.TrainModel(child_model,dataset_tool.StandardizeImages(x_deteriated), train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= dataset_tool.StandardizeImages(val_x), val_y=val_y)
                x_prediction_probs, x_predictions = child_model.Predict(dataset_tool.StandardizeImages(test_x), return_prediction_scores = True)
            else:
                training_stats = framework_tool.TrainModel(child_model,x_deteriated, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)
                x_prediction_probs, x_predictions = child_model.Predict(test_x, return_prediction_scores = True)


            for img_i in range(len(x_prediction_probs)):
                prediction_probs = [str(prob) for prob in list(x_prediction_probs[img_i])]
                prediction = x_predictions[img_i]

                deterioration_output_string += ",".join(prediction_probs) +","+str(prediction) +","+str(np.argmax(test_y[img_i])) +"\n"
                
            with open(deterioration_output_path, "w") as f:
                f.write(deterioration_output_string[:-1])
                
            #retest on test
            if(normalise_data):
                new_accuracy = child_model.EvaluateModel(dataset_tool.StandardizeImages(test_x), test_y, model_train_params["batch_size"])
            else:
                new_accuracy = child_model.EvaluateModel(test_x, test_y, model_train_params["batch_size"])
                
            print("metrics", model_instance.model.metrics_names)
            print("new accuracy",new_accuracy)

            test_results.append( (deterioration_rate*(deterioration_step+1),new_accuracy) )
            
        SaveExperimentResults(output_path,test_results)
