import numpy as np

import requests
import json

import cv2

import sys

import math

###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool
     

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
    #TODO: confirm taking the abs is correct
    return sorted(pixel_weight_list,key=lambda x: abs(x[2]),reverse=True)
 

def CreatePixelListForAllTrainSet(train_x, train_y, model_instance, explanation_instance,additional_args=None):
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
        }

    dataset_pixel_weight_lists = []

    total_imgs = len(train_x)
    #TODO: Could be parallelized
    for image_i in range(total_imgs):
        if(image_i % 100 == 0):
            print("Generating Explanation for Image: "+str(image_i)+ "/" + str(total_imgs))
            print("")
            
        image_x = train_x[image_i]
        explanation_image, explanation_text, prediction, additional_outputs = explanation_instance.Explain(image_x,additional_args=additional_args) 

        attribution_map =  np.array(additional_outputs["attribution_map"])
        pixel_weight_list = CreateOrderedPixelsList(attribution_map)

        dataset_pixel_weight_lists.append(pixel_weight_list)


    return dataset_pixel_weight_lists


def DeteriateImage(image,image_pixel_list,dataset_mean,deteriation_start_index,deteriation_end):
    deteriation_pixels = image_pixel_list[deteriation_start_index:deteriation_end]
    
    for px in deteriation_pixels:
        image[px[0]][px[1]] = [dataset_mean[0],dataset_mean[1],dataset_mean[2]]

    return image


def DeteriateDataset(Xs,deteriation_proportion,dataset_pixel_lists,deteriation_step,deteriation_index_step,dataset_mean):
    deteriation_start_index = deteriation_step*deteriation_index_step
    deteriation_end = min(len(dataset_pixel_lists[0])-1, (deteriation_step+1)*deteriation_index_step)
    
    modified_images = []
    
    total_imgs = len(Xs)
    #TODO: Could be parallelized
    for x_i in range(total_imgs):
        if(x_i % 100 == 0):
            print("Generating Explanation for Image: "+str(x_i)+ "/" + str(total_imgs))
            print("")
        new_image = DeteriateImage(Xs[x_i], dataset_pixel_lists[x_i],dataset_mean,deteriation_start_index,deteriation_end)
        modified_images.append(new_image)
    
    return np.array(modified_images)


def CreateChildModel(framework_tool, deteriation_proportion, model_train_params):
    model_save_path_suffix = "_"+model_train_params["experiment_id"]+"_"+str(deteriation_proportion+1)
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
    #dataset_name = "CIFAR-10"
    model_name = "vgg16_imagenet"
    explanation_name = "LRP"
    experiment_id="testROAR"
    output_path=str(experiment_id)+"_results.csv"

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
    train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source)
    print("num train examples: "+str(len(train_x)))


    #validate on up to 256 images only
    source = "validation"
    val_x, val_y = dataset_tool.GetBatch(batch_size = 512,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = 512,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num test examples: "+str(len(test_x)))
    
    
    print("calculating dataset mean")
    dataset_mean = dataset_tool.GetMean()
    print(dataset_mean)

    #INSTANTIATE MODEL
    model_train_params ={
    "learning_rate": 0.001
    ,"batch_size":128
    ,"num_train_steps":160
    ,"experiment_id":experiment_id
    }

    model_save_path_suffix = "_"+model_train_params["experiment_id"]+"_baseline"
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
    

    #INSTANTIATE EXPLANTION
    explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)



    #TRAIN MODEL
    training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)


    #INITAL TEST of MODEL
    test_results = [] # of the form: (proportion_of_deteriated_pixels, test_metrics) . test_metrics = [loss, accuracy] .


    baseline_accuracy = model_instance.EvaluateModel(test_x, test_y, model_train_params["batch_size"])
    print("metrics", model_instance.model.metrics_names)
    print("baseline_accuracy",baseline_accuracy)

    test_results.append((0,baseline_accuracy))

    #CREATE WORKING COPY OF TRAIN SET
    train_x_deteriated = np.copy(train_x)


    deteriation_rate = 0.05
    num_pixels = dataset_json["image_y"] * dataset_json["image_x"]
    deteriation_index_step = math.ceil(num_pixels * deteriation_rate)
    num_deteriation_steps = 20
    

    dataset_pixel_lists = CreatePixelListForAllTrainSet(train_x, train_y, model_instance, explanation_instance,additional_args=None)
    
    #TODO: Could be parallelized
    for deteriation_step in range(num_deteriation_steps):
        print("Starting Deteriation Step: "+str(deteriation_step))
        print("")
        #remove the deteriation_index_step * deteriation_step pixels from each image of train
        train_x_deteriated = DeteriateDataset(train_x_deteriated,deteriation_rate,dataset_pixel_lists,deteriation_step,deteriation_index_step,dataset_mean)

        #instantiate child model        
        child_model = CreateChildModel(framework_tool, deteriation_rate*(deteriation_step+1),model_train_params)
        
        #retrain on modified train
        training_stats = framework_tool.TrainModel(child_model,train_x_deteriated, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)

        #retest on test
        new_accuracy = child_model.EvaluateModel(test_x, test_y, model_train_params["batch_size"])
        print("metrics", model_instance.model.metrics_names)
        print("baseline_accuracy",new_accuracy)

        test_results.append( (deteriation_rate*(deteriation_step+1),new_accuracy) )
        
    SaveExperimentResults(output_path,test_results)

    # cv2.imshow("test_image",train_x_deteriated[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
