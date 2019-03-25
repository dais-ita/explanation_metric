import sys
import os

import math

import numpy as np

try:
    from PIL import Image
except:
    from PILLOW import Image

import cv2

import tensorflow as tf
import random

from ROAR_pipeline import CreatePixelListForAllData, LoadPixelListFromPath, SavePixelList, PerturbData, CreateConstantPeturbFunction, SaveExperimentResults, CreateOrderedPixelsList, GenerateRandomPixelWeights

# INITIALISE FRAMEWORK
###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
# framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool


np.random.seed(42)
tf.set_random_seed(1234)
random.seed(1234)

def SaveImage(image, output_path, flip_channels = False):
    output_image = image
    if(flip_channels):
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    if(output_image.max() <= 1):
        output_image = output_image*255
    im = Image.fromarray(output_image.astype(np.uint8))
    im.save(output_path)

if __name__ == "__main__":
    #INITIALISE EXPERIMENT PARAMETERS    
    dataset_name = "Traffic Congestion Image Classification (Resized)"
    dataset_name = "CIFAR-10"
    
    # model_name = "inception_v3_imagenet"
    model_name = "vgg16"
    normalise_data = True

    explanation_names = ["LIME","Shap","random"]
    load_from_pixel_list_path_dict={
        "LIME": os.path.join("pixel_lists","TEST_CIFAR-10_LIME_1553397515.pkl")
        ,"Shap": os.path.join("pixel_lists","TEST_CIFAR-10_Shap_1553406978.pkl")
        ,"random": os.path.join("pixel_lists","TEST_CIFAR-10_random_1553390622.pkl")
    }
    explanation_name = "LIME"
    experiment_id="testPRESERVATION_"+dataset_name
    output_path=os.path.join("results",str(experiment_id)+"_"+explanation_name+"_results.csv")
    load_base_model_if_exist = True
    save_pixel_list = True

    use_deletion_game = False
    deterioration_rate = 0.05
    num_deterioration_steps = 20

    save_deteriorated_images_below_index = 5
    
    generate_random_pixel_list = False
    if(explanation_name == "random"):
        generate_random_pixel_list = True
    
    load_from_pixel_list_path = "" #os.path.join("pixel_lists",pixel_list_name) "TEST_CIFAR-10_random_1553144498.pkl" #"TEST_CIFAR-10_LIME_1552602173.pkl" #make sure to load test set

    model_train_params ={
    "learning_rate": 0.001
    ,"batch_size":128
    ,"num_train_steps":150
    ,"experiment_id":experiment_id
    ,"dropout":0.5
    }

    
    

    
    
    


    
# INITIALISE DATASET, MODEL and EXPLANATION
    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)

    ##DATASET
    dataset_json, dataset_tool = framework_tool.LoadFrameworkDataset(dataset_name, load_split_if_available = True)

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


    standardized_train_x = dataset_tool.StandardizeImages(train_x)

    #validate on up to 256 images only
    source = "validation"
    val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
    print("num test examples: "+str(len(test_x)))
    

    print("calculating dataset mean")
    dataset_mean = dataset_tool.GetMean()
    print(dataset_mean)

    if(normalise_data):
        denormalise_function = dataset_tool.CreateDestandardizeFuntion()


    #INSTANTIATE MODEL
    model_save_path_suffix = ""
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args =model_train_params)
    

    


    #TRAIN OR LOAD MODEL
    model_load_path = model_instance.model_dir
    if(os.path.exists(model_load_path) == True and load_base_model_if_exist == True):
        model_instance.LoadModel(model_load_path)
    else:
        if(normalise_data):
            training_stats = framework_tool.TrainModel(model_instance,dataset_tool.StandardizeImages(train_x), train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= dataset_tool.StandardizeImages(val_x), val_y=val_y)

        else:
            training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)

    #FOR EACH EXPLANATION
    for explanation_name in explanation_names:
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
        
        
        ## INITALISE DETERIORATION IMAGES
        x_deteriated = np.copy(test_x)

        #RESIZE IMAGES IF NEEDED
        #Images may need resizing for model. If that's the case, the framework would have done this automatically during training and explanation generations. 
        #we need to do this manually before deteriation step, the framework model class can do this for us. 
        x_deteriated = model_instance.CheckInputArrayAndResize(x_deteriated,model_instance.min_height,model_instance.min_width)

        num_pixels = dataset_json["image_y"] * dataset_json["image_x"]
        
        num_pixels_in_padded = x_deteriated[0].shape[:-1][0] * x_deteriated[0].shape[:-1][1]
        deterioration_index_step = int(math.ceil(num_pixels_in_padded * deterioration_rate))
    #EXPLAIN and PRODUCE ORDERED PIXEL LISTS
        pixel_lists = []
        
        if(generate_random_pixel_list):
            print("Generating Random Pixel List")
            pixel_lists = GenerateRandomPixelWeights(x_deteriated.shape)
            if(save_pixel_list):
                    print("saving pixel lists")
                    SavePixelList(experiment_id,explanation_name,pixel_lists)

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
                    SavePixelList(experiment_id,explanation_name,pixel_lists)


    # EVALUATION
        
        perturbation_function = CreateConstantPeturbFunction(dataset_mean)

        
        #For each level of deterioration:
        for deterioration_step in range(num_deterioration_steps):
            deterioration_output_path = str(experiment_id)+"_"+explanation_name+"_"+format(deterioration_step, '05d')+"_deterioration_results.csv"
            deterioration_output_path = os.path.join("results","image_results",deterioration_output_path)
            
            deterioration_output_string = ""
            print("")
            print("______")
            print("Starting deterioration Step: "+str(deterioration_step))
            print("")
    
            #DELETE OR KEEP PHASE
            x_deteriated = PerturbData(x_deteriated,deterioration_rate,pixel_lists,deterioration_step,deterioration_index_step,perturbation_function, deletion_game=use_deletion_game, total_num_deterioration_steps=num_deterioration_steps)
            
            #RETEST PHASE
            x_prediction_probs, x_predictions = model_instance.Predict(dataset_tool.StandardizeImages(x_deteriated), return_prediction_scores = True)
            
            for img_i in range(len(x_prediction_probs)):
                prediction_probs = [str(prob) for prob in list(x_prediction_probs[img_i])]
                prediction = x_predictions[img_i]

                deterioration_output_string += ",".join(prediction_probs) +","+str(prediction) +","+str(np.argmax(test_y[img_i])) +"\n"
                
                if(img_i < save_deteriorated_images_below_index):
                    save_output_path = os.path.join("deteriorated_images", experiment_id + "_" +explanation_name+"_"+ str(deterioration_step) + "_" + str(img_i) +".jpg")
                    SaveImage(x_deteriated[img_i],save_output_path)
            with open(deterioration_output_path, "w") as f:
                f.write(deterioration_output_string[:-1])

            if(normalise_data):
                new_accuracy = model_instance.EvaluateModel(dataset_tool.StandardizeImages(x_deteriated), test_y, model_train_params["batch_size"])
            else:
                new_accuracy = model_instance.EvaluateModel(x_deteriated, test_y, model_train_params["batch_size"])
            
            print("metrics", model_instance.model.metrics_names)
            print("new accuracy",new_accuracy)

            test_results.append( (deterioration_rate*(deterioration_step+1),new_accuracy) )
            
        SaveExperimentResults(output_path,test_results)

