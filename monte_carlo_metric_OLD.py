import sys
import os

import random

import pickle

import numpy as np
import tensorflow as tf

from ROAR_pipeline import LoadPixelListFromPath
from retest_pipeline import SaveImage

#seed setting
np.random.seed(42)
tf.set_random_seed(1234)
random.seed(1234)


# INITIALISE FRAMEWORK
###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
# framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool


explanation_pixel_lists_dict = {
        "pixel_lists":{
            "LIME":{
                "test":{
                    "CIFAR-10":"TEST_CIFAR-10_LIME_1553397515.pkl"
                    # "CIFAR-10":"TEST_CIFAR-10_LIME_1553397515_SMALL.pkl"
                },
                "train":{
                    "CIFAR-10":"TRAIN_testROAR_CIFAR-10_LIME_1553461654.pkl"
                }
            },
            "Shap":{
                "test":{
                    "CIFAR-10":"TEST_CIFAR-10_Shap_1553406978.pkl"
                },
                "train":{
                    "CIFAR-10":"TRAIN_testROAR_CIFAR-10_Shap_1553686507.pkl"
                }
            },
            "random":{
                "test":{
                    "CIFAR-10":"TEST_CIFAR-10_random_1553390622.pkl"
                },
                "train":{
                    "CIFAR-10":"TRAIN_testROAR_CIFAR-10_random_1553734921.pkl"
                }
            }
        }
    }

def GetClassImageIndexs(class_names,y_vals):
    class_images = {}
    for class_name_i, class_name in enumerate(class_names):
        class_images[class_name] = [i for i in range(len(test_y)) if np.argmax(test_y[i]) == class_name_i]
    
    return class_images


def GetCorrectPredictionIndexs(predictions,truth_indexs):
    return [i for i in range(len(predictions)) if predictions[i] == np.argmax(truth_indexs[i]) ]    


def NormalisePixelList(image_pixel_list):
    p_min = min([p[2] for p in image_pixel_list])

    positive_list = []
    
    total = 0
    for p in image_pixel_list:
        positive_list.append([p[0],p[1],p[2]+p_min])
        total += (p[2]+p_min)

    distribution_list = []
    for p in positive_list:
        distribution_list.append([p[0],p[1],p[2]/float(total)])
    

    return distribution_list
    



def find_interval(x, partition):
    """ find_interval -> i
        partition is a sequence of numerical values
        x is a numerical value
        The return value "i" will be the index for which applies
        partition[i] < x < partition[i+1], if such an index exists.
        -1 otherwise
    """
    
    for i in range(0, len(partition)):
        if x < partition[i]:
            return i-1
    return -1


def weighted_choice(sequence,weights):
    """ 
    weighted_choice selects a random element of 
    the sequence according to the list of weights
    """
    
    x = np.random.random()
    
    cum_weights = [0] + list(np.cumsum(weights))
    index = find_interval(x, cum_weights)
    return sequence[index]



def weighted_sample(population, weights, k):
    """ 
    This function draws a random sample of length k 
    from the sequence 'population' according to the 
    list of weights
    """
    sample = set()
    population = list(population)
    weights = list(weights) 
    while len(sample) < k:
        choice = weighted_choice(population, weights)
        sample.add(tuple(choice))
        index = population.index(choice)
        weights.pop(index)
        population.remove(choice)
        new_sum = sum(weights)
        weights = [ x / new_sum for x in weights]
    return list(sample)




def SampleFromPixelList(normalised_pixel_list,num_pixels_to_select):
    weights = [p[2] for p in normalised_pixel_list]
    return weighted_sample(normalised_pixel_list, weights, num_pixels_to_select)


def DeteriorateImage(working_image,sampled_pixels,dataset_mean):
    for px in sampled_pixels:
        working_image[px[0]][px[1]] = [dataset_mean[0],dataset_mean[1],dataset_mean[2]]
    
    return working_image

if __name__ == "__main__":
    dataset_name = "CIFAR-10"
    model_name = "vgg16" #M
    test_or_train_data = "test"
    normalise_data = True

    deterioration_rate = 0.05
    num_deterioration_steps = 5

    num_monte_carlo_samples = 10
    
    correct_predictions_only = True

    experiment_id = "monte_carlo_metric_"+dataset_name+"_"+model_name+"_"+test_or_train_data
    if(correct_predictions_only):
        experiment_id += "_correct_only"

    explanation_names = ["LIME","Shap","random"] #E

    load_from_pixel_list_path_dict={}
    for explanation_name in explanation_names:
        load_from_pixel_list_path_dict[explanation_name] = os.path.join("pixel_lists",explanation_pixel_lists_dict["pixel_lists"][explanation_name][test_or_train_data][dataset_name])
   
    model_train_params ={
    "learning_rate": 0.001
    ,"batch_size":128
    ,"num_train_steps":150
    ,"experiment_id":experiment_id
    ,"dropout":0.5
    }

    save_deteriorated_images_below_image_index = 5
    save_deteriorated_images_below_sample_index = 5

    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)

    ##DATASET
    dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name, load_split_if_available = True)

    label_names = [label["label"] for label in dataset_json["labels"]] # gets all labels in dataset. To use a subset of labels, build a list manually

    #LOAD DATA
    #load all train images as model handles batching
    # print("load training data")
    # print("")
    # source = "train"
    # train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source, shuffle=False)
    # print("num train examples: "+str(len(train_x)))


    # standardized_train_x = dataset_tool.StandardizeImages(train_x)

    # #validate on up to 256 images only
    # source = "validation"
    # val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    # print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
    print("num test examples: "+str(len(test_x)))
    

    print("calculating dataset mean")
    dataset_mean = dataset_tool.GetMean()
    print(dataset_mean)

    if(normalise_data):
        denormalise_function = dataset_tool.CreateDestandardizeFuntion()

    dataset_class_names = dataset_tool.one_hot_list
    class_images_indexs = GetClassImageIndexs(dataset_class_names,test_y)

    #INSTANTIATE MODEL
    model_save_path_suffix = ""
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args =model_train_params)


    #LOAD MODEL
    model_load_path = model_instance.model_dir
    model_instance.LoadModel(model_load_path) #M (loaded trained model)


    x_deteriated = np.copy(test_x)
    x_prediction_probs, x_predictions = model_instance.Predict(dataset_tool.StandardizeImages(x_deteriated), return_prediction_scores = True)

    correct_prediction_indexs = set(GetCorrectPredictionIndexs(x_predictions,test_y)) 

    #RESIZE IMAGES IF NEEDED
    #Images may need resizing for model. If that's the case, the framework would have done this automatically during training and explanation generations. 
    #we need to do this manually before deteriation step, the framework model class can do this for us. 
    x_deteriated = model_instance.CheckInputArrayAndResize(x_deteriated,model_instance.min_height,model_instance.min_width)

    for explanation_name in explanation_names[:1]:
        print("Starting Explanation: "+explanation_name)
        #Load Importance Pickle
        print("Loading Pixel List")
        pixel_list = LoadPixelListFromPath(load_from_pixel_list_path_dict[explanation_name])
        
        explanation_deterioration_results = {}
        
        for class_name_i,class_name in enumerate(dataset_class_names):
            print("Begining Class: "+class_name)
            # if(class_name!= "horse"):
            #     continue
            
            explanation_deterioration_results[class_name] = []

            #Get Class Image Indexs
            class_image_indexs = class_images_indexs[class_name]

            #Filter to correct predictions only
            if(correct_predictions_only):
                class_image_indexs = list( correct_prediction_indexs.intersection(set(class_image_indexs)) )
            
            for img_i in class_image_indexs[:]:
                print("Begining Img Index: "+str(img_i))
                image_results = {"img_i":img_i,"results":[]}
                image_pixel_list = pixel_list[img_i]
                original_prediction_index = x_predictions[img_i]
                original_prediction_strength = x_prediction_probs[img_i][original_prediction_index]
                #Normalise Importance Array
                normalised_pixel_list = NormalisePixelList(image_pixel_list)

                pixel_list_length = len(image_pixel_list)
                for current_deterioration_proportion in np.arange(deterioration_rate,deterioration_rate*(num_deterioration_steps+1),deterioration_rate):
                    print("Begining Deterioration Proportion: "+str(current_deterioration_proportion))
                    num_pixels_to_select = current_deterioration_proportion * pixel_list_length
                    proportion_results = {"proportion": current_deterioration_proportion,"results":[]}

                    for sample_i in range(num_monte_carlo_samples):
                        #Make Random Selections of deteioration proportion number of Pixels based on Normalised Importance
                        sampled_pixels = SampleFromPixelList(normalised_pixel_list,num_pixels_to_select)
                        
                        #Detiorate Image
                        working_image = np.copy(x_deteriated[img_i])
                        print("Detiorating Image")
                        deteriorated_image = DeteriorateImage(working_image,sampled_pixels,dataset_mean)
                        
                        if(img_i < save_deteriorated_images_below_image_index and sample_i < save_deteriorated_images_below_sample_index):
                            save_output_path = os.path.join("sample_deteriorated_images", experiment_id + "_" +explanation_name+"_"+ str(img_i) + "_" + str(current_deterioration_proportion) + "_" + str(sample_i) +".jpg")
                            SaveImage(deteriorated_image,save_output_path)
                        
                        
                        det_x_prediction_probs, det_x_predictions = model_instance.Predict(dataset_tool.StandardizeImages(np.array([deteriorated_image])), return_prediction_scores = True)


                        #Store deterioration
                        proportion_results["results"].append(det_x_prediction_probs[0][original_prediction_index]/float(original_prediction_strength))
                    
                    image_results["results"].append(proportion_results)

                    #store image vector

                    explanation_deterioration_results[class_name].append(image_results)

        #output image vectors
        print("Outputing Vector Dict")
        output_vector_dict_path = os.path.join("carlo_vector_dicts",experiment_id+"_"+explanation_name+".pkl")

        with open(output_vector_dict_path,"wb") as f:
            pickle.dump(explanation_deterioration_results,f)

