import sys
import os

# INITIALISE FRAMEWORK
###UPDATE FRAMEWORK PATH
framework_path = "/media/harborned/ShutUpN/repos/dais/interpretability_framework"
framework_path = "/Users/harborned/Documents/repos/dais/interpretability_framework"

sys.path.append(framework_path)

from DaisFrameworkTool import DaisFrameworkTool



if __name__ == "__main__":
    #INITIALISE EXPERIMENT PARAMETERS    
    dataset_name = "Traffic Congestion Image Classification (Resized)"
    dataset_name = "CIFAR-10"
    model_name = "vgg16_imagenet"
    explanation_name = "LRP"
    experiment_id="testRETRAIN_"+dataset_name
    output_path=str(experiment_id)+"_"+explanation_name+"_results.csv"
    load_base_model_if_exist = True
    save_pixel_list = True

    deteriation_rate = 0.05
    num_deteriation_steps = 20
    
    load_from_pixel_list_path = ""

    
    


    framework_tool = DaisFrameworkTool(explicit_framework_base_path=framework_path)

# INITIALISE DATASET, MODEL and EXPLANATION
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
    val_x, val_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num validation examples: "+str(len(val_x)))


    #load train data
    source = "test"
    test_x, test_y = dataset_tool.GetBatch(batch_size = 256,even_examples=True, y_labels_to_use=label_names, split_batch = True,split_one_hot = True, batch_source = source)
    print("num test examples: "+str(len(test_x)))
    
    
    print("calculating dataset mean")
    dataset_mean = dataset_tool.GetMean()
    print(dataset_mean)

    #INSTANTIATE MODEL
    model_train_params ={
    "learning_rate": 0.0001
    ,"batch_size":128
    ,"num_train_steps":200
    ,"experiment_id":experiment_id
    }

    model_save_path_suffix = "_"+model_train_params["experiment_id"]+"_baseline"
    model_instance = framework_tool.InstantiateModelFromName(model_name,model_save_path_suffix,dataset_json,additional_args = {"learning_rate":model_train_params["learning_rate"]})
    

    #INSTANTIATE EXPLANTION
    explanation_instance = framework_tool.InstantiateExplanationFromName(explanation_name,model_instance)



    #TRAIN OR LOAD MODEL
    model_load_path = model_instance.model_dir
    if(os.path.exists(model_load_path) == True and load_base_model_if_exist == True):
        model_instance.LoadModel(model_load_path)
    else:
        training_stats = framework_tool.TrainModel(model_instance,train_x, train_y, model_train_params["batch_size"], model_train_params["num_train_steps"], val_x= val_x, val_y=val_y)



#INITAL TEST of MODEL
    test_results = [] # of the form: (proportion_of_deteriated_pixels, test_metrics) . test_metrics = [loss, accuracy] .

    baseline_accuracy = model_instance.EvaluateModel(test_x, test_y, model_train_params["batch_size"])
    print("metrics", model_instance.model.metrics_names)
    print("baseline_accuracy",baseline_accuracy)

    test_results.append((0,baseline_accuracy))

#EXPLAIN and PRODUCE ORDERED PIXEL LISTS


# EVALUATION
    #For each level of deterioration:
        #DELETE OR KEEP PHASE
        #for each image in test images:
            #deterioate image

        #RETEST PHASE