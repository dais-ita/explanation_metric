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

from experiment_param_dict import param_dict

from ROAR_pipeline import CreatePixelListForAllData, LoadPixelListFromPath, SavePixelList, CreateConstantPeturbFunction, SaveExperimentResults, CreateOrderedPixelsList

# INITIALISE FRAMEWORK
###UPDATE FRAMEWORK PATH
framework_path = "/home/richard/git/interpretability_framework"
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

def GenerateSameRandomPixelWeights(images_shape, rseed):
    pixel_weight_size = list(images_shape[1:])
    pixel_weight_size[-1] = 1
    
    dataset_pixel_weight_lists = []
    random_generator = np.random.RandomState(rseed)
    pix_weights = random_generator.uniform(size=pixel_weight_size)

    for image_i in range(images_shape[0]):
        if(image_i % 100 == 0):
            print("Generating Random Pixel List for:" + str(image_i))
        pixel_weight_list = CreateOrderedPixelsList(pix_weights)
        dataset_pixel_weight_lists.append(pixel_weight_list)
        
    return dataset_pixel_weight_lists

def DeteriorateImageWithRandomColour(img,x,y,rseed):
    random_generator = np.random.RandomState(rseed)
    img[x][y] = [random_generator.rand(),random_generator.rand(),random_generator.rand()]
    
    return img

def CreateGridPerturbationFunction(grid_width=3,grid_height=3, pixel_operation_function=DeteriorateImageWithRandomColour):
    grid_width_distance = int((grid_width-1) / 2)
    grid_height_distance = int((grid_height-1) / 2)
    print(grid_height_distance)
    def DeteriorateGridOfImageWithRandomColour(img,x,y,rseed):
        for width_modifier in range(-grid_width_distance,(grid_width_distance+1),1):
            random_generator = np.random.RandomState(rseed)
            for height_modifier in range(-grid_height_distance,(grid_height_distance+1),1):
                if (x+width_modifier>=0) and (x+width_modifier<=img.shape[0]) and (y+height_modifier>=0) and (y+height_modifier<=img.shape[1]):
                    img[x+width_modifier][y+height_modifier] = [random_generator.rand(),random_generator.rand(),random_generator.rand()]
                    # img = pixel_operation_function(img, x+width_modifier,y+height_modifier)
        return img

    return DeteriorateGridOfImageWithRandomColour

#PERTURBATION FUNCTIONS
def PerturbImage(image,image_pixel_list,perturbation_function,deterioration_start_index,deterioration_end,rseed):
    #generic perturbation mangement function that takes a specific perturbation function as an argument
    deterioration_pixels = image_pixel_list[deterioration_start_index:deterioration_end]
    
    for idx, px in enumerate(deterioration_pixels):
        if rseed is None:
            image = perturbation_function(image, px[0], px[1])
        else:
            image = perturbation_function(image, px[0], px[1], rseed+idx)
        
    return image

def PerturbData(Xs,deterioration_proportion,dataset_pixel_lists,deterioration_step,deterioration_index_step,perturbation_function, deletion_game=True, total_num_deterioration_steps=20, rseed=None):
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
        new_image = PerturbImage(Xs[x_i], dataset_pixel_lists[x_i],perturbation_function,deterioration_start_index,deterioration_end,rseed)
        modified_images.append(new_image)
    
    return np.array(modified_images)

if __name__ == "__main__":
    param_set_name = sys.argv[1]
    print("PARAM SET: " + param_set_name)
    params = param_dict[param_set_name]

    #INITIALISE EXPERIMENT PARAMETERS    
    dataset_name = params["dataset_name"]
    model_name = params["model_name"]
    normalise_data = params["normalise_data"]
    explanation_names = params["explanation_names"]
    load_from_pixel_list_path_dict = params["load_from_pixel_list_path_dict"]
    perturb_method = params["perturb_method"]
    experiment_id = params["experiment_id"]
    use_deletion_game = params["use_deletion_game"]
    load_base_model_if_exist = params["load_base_model_if_exist"]
    save_pixel_list = params["save_pixel_list"]
    random_seed = params["random_seed"]
    explicit_pixels_per_step = params["explicit_pixels_per_step"]
    deterioration_rate = params["deterioration_rate"]

    if explicit_pixels_per_step is not None:
        experiment_id += str(explicit_pixels_per_step) + "px_"

    if(use_deletion_game):
        experiment_id += "deletion_game"
    else:
        experiment_id += "preservation_game"

    experiment_id+="_"+dataset_name+"_"+perturb_method

    if explicit_pixels_per_step is None:
        num_deterioration_steps = int(1./deterioration_rate)
    else:
        num_deterioration_steps = 100

    add_random = ""
    if random_seed is not None:
        add_random += "_"+str(random_seed)

    save_deteriorated_images_below_index = 10
    
    model_train_params ={
    "learning_rate": 0.001
    ,"batch_size":64
    ,"num_train_steps":250
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
    #TODO change batch sizes to -1 , 256 , -1
    train_x, train_y = dataset_tool.GetBatch(batch_size = -1,even_examples=False, y_labels_to_use=label_names, split_batch = True, split_one_hot = True, batch_source = source, shuffle=False)
    print("num train examples: "+str(len(train_x)))


    #standardized_train_x = dataset_tool.StandardizeImages(train_x)

    #validate on up to 256 images only
    #source = "validation"
    #val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=False, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    #print("num validation examples: "+str(len(val_x)))


    #load test data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = -1,even_examples=False, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source, shuffle=False)
    print("num test examples: "+str(len(test_x)))
    

    print("loading train dataset mean")
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
        raise Exception("Model does not exist: " + model_load_path)

    #FOR EACH EXPLANATION
    for explanation_name in explanation_names:
        print("Generating Results for: " + explanation_name+add_random)
        load_from_pixel_list_path = ""
        
        if(explanation_name in load_from_pixel_list_path_dict):
            load_from_pixel_list_path = load_from_pixel_list_path_dict[explanation_name]
        
        output_path=os.path.join("results",str(experiment_id)+"_"+explanation_name+add_random+"_results.csv")
        
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
        
        ## INITALISE DETERIORATION IMAGES
        x_deteriated = np.copy(test_x)

        #RESIZE IMAGES IF NEEDED
        #Images may need resizing for model. If that's the case, the framework would have done this automatically during training and explanation generations. 
        #we need to do this manually before deteriation step, the framework model class can do this for us. 
        x_deteriated = model_instance.CheckInputArrayAndResize(x_deteriated,model_instance.min_height,model_instance.min_width)

        num_pixels = dataset_json["image_y"] * dataset_json["image_x"]
        
        num_pixels_in_padded = x_deteriated[0].shape[:-1][0] * x_deteriated[0].shape[:-1][1]
        if(explicit_pixels_per_step is None):
            deterioration_index_step = int(math.ceil(num_pixels_in_padded * deterioration_rate))
        else:
            deterioration_index_step = explicit_pixels_per_step

    #EXPLAIN and PRODUCE ORDERED PIXEL LISTS
        pixel_lists = []
        
        if(generate_random_pixel_list):
            print("Generating Random Pixel List")
            pixel_lists = GenerateSameRandomPixelWeights(x_deteriated.shape, random_seed)
            if(save_pixel_list):
                    print("saving pixel lists")
                    SavePixelList(experiment_id,explanation_name+add_random,pixel_lists)

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
                    SavePixelList(experiment_id,explanation_name+add_random,pixel_lists)


    # EVALUATION
        
        ### SELECT PERTURBATION FUNCTION ####
        if(perturb_method == "grid"):
            perturbation_function = CreateGridPerturbationFunction()
        elif(perturb_method == "mean"):
            perturbation_function = CreateConstantPeturbFunction(dataset_mean)
        elif(perturb_method == "random"):
            perturbation_function = DeteriorateImageWithRandomColour
        else:
            print("Perturb method not recognised")
            assert False
        paper_results_dict = {}

        ###original prediction strengths

        x_prediction_probs, x_predictions = model_instance.Predict(dataset_tool.StandardizeImages(x_deteriated), return_prediction_scores = True)
            
        for img_i in range(len(x_prediction_probs)):
            paper_results_dict[img_i] = {"results":[],"ground_truth":np.argmax(test_y[img_i]),"original_prediction":x_predictions[img_i]}
            original_prediction_i = paper_results_dict[img_i]["original_prediction"]
            paper_results_dict[img_i]["results"].append(x_prediction_probs[img_i][original_prediction_i])

        #For each level of deterioration:
        for deterioration_step in range(num_deterioration_steps):
            deterioration_output_path = str(experiment_id)+"_"+explanation_name+add_random+"_"+format(deterioration_step, '05d')+"_deterioration_results.csv"
            deterioration_output_path = os.path.join("results","image_results",deterioration_output_path)
            


            deterioration_output_string = ""
            print("")
            print("______")
            print("Starting deterioration Step: "+str(deterioration_step))
            print("")
    
            #DELETE OR KEEP PHASE
            rs = None
            if perturb_method != "mean":
                rs = (deterioration_step * 159) + 951
            x_deteriated = PerturbData(x_deteriated,deterioration_rate,pixel_lists,deterioration_step,deterioration_index_step,perturbation_function, deletion_game=use_deletion_game, total_num_deterioration_steps=num_deterioration_steps, rseed=rs)
            
            #RETEST PHASE
            x_prediction_probs, x_predictions = model_instance.Predict(dataset_tool.StandardizeImages(x_deteriated), return_prediction_scores = True)
            
            for img_i in range(len(x_prediction_probs)):
                prediction_probs = [str(prob) for prob in list(x_prediction_probs[img_i])]
                prediction = x_predictions[img_i]

                deterioration_output_string += ",".join(prediction_probs) +","+str(prediction) +","+str(np.argmax(test_y[img_i])) +"\n"
                
                if(img_i < save_deteriorated_images_below_index):
                    save_output_path = os.path.join("deteriorated_images", experiment_id + "_" +explanation_name+add_random+"_"+ str(img_i) + "_" + str(deterioration_step) +".png")
                    SaveImage(x_deteriated[img_i],save_output_path)
                
                ###paper results
                original_prediction_i = paper_results_dict[img_i]["original_prediction"]
                paper_results_dict[img_i]["results"].append(x_prediction_probs[img_i][original_prediction_i])
                    

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

        paper_results_output_path = str(experiment_id)+"_"+explanation_name+add_random+"_results.csv"
        paper_results_output_path = os.path.join("paper_results",paper_results_output_path)
        paper_results_string = ",".join(["explanation_name"
        ,"ground_truth"
        ,"original_prediction_i"
        ,"img_i"
        ,"trial_i"])

        if(explicit_pixels_per_step is None):
            for deterioration_step_track in range(num_deterioration_steps+1): 
                paper_results_string += "," + str(deterioration_rate*deterioration_step_track)
        else:
            for deterioration_step_track in range(num_deterioration_steps+1): 
                paper_results_string += "," + str(explicit_pixels_per_step*deterioration_step_track)
                
        
        for img_i in range(len(x_prediction_probs)):
            res_string = ",".join( [str(v) for v in paper_results_dict[img_i]["results"] ] )
            paper_results_string += "\n"
            paper_results_string += ",".join(str(v) for v in [
                explanation_name+add_random
                ,paper_results_dict[img_i]["ground_truth"]
                ,paper_results_dict[img_i]["original_prediction"]
                ,img_i
                ,0
                ,res_string
            ])
        
        with open(paper_results_output_path, "w") as f:
                f.write(paper_results_string)