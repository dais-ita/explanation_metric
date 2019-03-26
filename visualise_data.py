import os
import pickle

import matplotlib.pyplot as plt
from cycler import cycler

import numpy as np 



#visualisations
# [X] X=detrioration_step, y=  one explanation technique - one image - 10 rates of prediction stength for each class (mark the correct class and the original prediction)
# [X] X=detrioration_step, y=  each explanation technique - one image - one prediction stength for original class
# [ ] X=detrioration_step, y=  each explanation technique - aggreagte images - one average of percentage of prediction stength for original class
# [X] X=detrioration_step, y=  each explanation technique - aggreagte images - one testset accuracy

#"testRETEST_CIFAR-10_LIME_00000_deteriation_results.csv"


def DisplayPredictionStrengthsAcrossAllClassesForOneExplanationOneImage(explanation_name, image_i, image_results_dict):
    image_results = image_results_dict[explanation_name][image_i]["results"]
    predicted_class = image_results_dict[explanation_name][image_i]["original_prediction"]
    ground_truth_class = image_results_dict[explanation_name][image_i]["ground_truth"]
    
    num_classes = len(image_results[0])

    class_results = [ [] for i in range(num_classes) ]

    detrioration_steps = list(range(0,20))

    for detrioration_step in detrioration_steps:
        for class_i in range(num_classes):
            class_results[class_i].append(image_results[detrioration_step][class_i])
    

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    
    colormap = plt.cm.gist_ncar
    ax1.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, num_classes)])
    
    ax1.set_title('Explanation: '+explanation_name+' Image Index: '+str(image_i) + ' Original Prediction Index: '+str(predicted_class) + ' Correct Class Index: '+str(ground_truth_class))
    
    for class_i in range(num_classes):
        ax1.plot(detrioration_steps,class_results[class_i], label="Class: "+str(class_i))
    
    plt.legend(loc='upper right')
    plt.show()


def DisplayPredictionStrengthOfPredicitedClassForAllExplanationOneImage(explanation_names, image_i, image_results_dict):
    explanation_results = {}

    for explanation_name in explanation_names:
        explanation_results[explanation_name] = []

        image_results = image_results_dict[explanation_name][image_i]["results"]
        predicted_class = image_results_dict[explanation_name][image_i]["original_prediction"]
        ground_truth_class = image_results_dict[explanation_name][image_i]["ground_truth"]
        
        num_classes = len(image_results[0])

        
        detrioration_steps = list(range(0,20))

        for detrioration_step in detrioration_steps:
            explanation_results[explanation_name].append(image_results[detrioration_step][predicted_class])
        

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    
    colormap = plt.cm.gist_ncar
    ax1.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, num_classes)])
    
    ax1.set_title('Image Index: '+str(image_i) + ' Original Prediction Index: '+str(predicted_class) + ' Correct Class Index: '+str(ground_truth_class))
    
    for explanation_name in explanation_names:
        ax1.plot(detrioration_steps,explanation_results[explanation_name], label="Explanation: "+str(explanation_name))
    
    plt.legend(loc='upper right')
    plt.show()


def DisplayTestAccuraciesForAllClassesForAllExplanationsAcrossAllImages(explanation_names,accuracies_dict):
    detrioration_steps = list(range(0,21))

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    
    colormap = plt.cm.gist_ncar
    ax1.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, len(explanation_names))])
    
    ax1.set_title("Degredatation of Test Accuracy")
    
    for explanation_name in explanation_names:
        ax1.plot(detrioration_steps,accuracies_dict[explanation_name], label="Explanation: "+str(explanation_name))
    
    plt.legend(loc='upper right')
    plt.show()



def DisplayOriginalPredictionDegradationForAllExplanationsAcrossAllImages(explanation_names, image_results_dict):
    detrioration_steps = list(range(0,20))

    # predicted_class = image_results_dict[explanation_names[0]][image_i]["original_prediction"]
    # original_strength = image_results_dict[explanation_names[0]][image_i]["original_prediction_score"]
    
    # ground_truth_class = image_results_dict[explanation_names[0]][image_i]["ground_truth"]
    
    explanation_all_results = {}
    explanation_results = {}

    for explanation_name in explanation_names:
        explanation_all_results[explanation_name] = []

        for i in range(len(image_results_dict[explanation_name][0]["original_prediction_degradations"])):
            explanation_all_results[explanation_name].append([])
        
        for image_i in range(len(image_results_dict[explanation_name])):
            for i in range(len(image_results_dict[explanation_name][image_i]["original_prediction_degradations"])):
                explanation_all_results[explanation_name][i].append(image_results_dict[explanation_name][image_i]["original_prediction_degradations"][i])
        
        explanation_results[explanation_name] = [np.mean(i) for i in explanation_all_results[explanation_name] ]
        

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    
    colormap = plt.cm.gist_ncar
    ax1.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, len(explanation_names))])
    
    ax1.set_title("Average Degradation Across Time Step for Entire Testset")
    
    for explanation_name in explanation_names:
        ax1.plot(detrioration_steps,explanation_results[explanation_name], label="Explanation: "+str(explanation_name))
    
    plt.legend(loc='upper right')
    plt.show()


def DisplayOriginalPredictionDegradationForAllExplanationsForOneImage(explanation_names, image_i, image_results_dict):
    detrioration_steps = list(range(0,20))

    predicted_class = image_results_dict[explanation_names[0]][image_i]["original_prediction"]
    original_strength = image_results_dict[explanation_names[0]][image_i]["original_prediction_score"]
    
    ground_truth_class = image_results_dict[explanation_names[0]][image_i]["ground_truth"]
    
    explanation_results = {}

    for explanation_name in explanation_names:
        explanation_results[explanation_name] = image_results_dict[explanation_name][image_i]["original_prediction_degradations"]
        

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    
    colormap = plt.cm.gist_ncar
    ax1.set_prop_cycle(color=[colormap(i) for i in np.linspace(0, 0.9, len(explanation_names))])
    
    ax1.set_title('Image Index: '+str(image_i) + ' Original Prediction Index: '+str(predicted_class) + ' Original Prediction Score: '+str(original_strength)  + ' Correct Class Index: '+str(ground_truth_class))
    
    for explanation_name in explanation_names:
        ax1.plot(detrioration_steps,explanation_results[explanation_name], label="Explanation: "+str(explanation_name))
    
    plt.legend(loc='upper right')
    plt.show()


def GetAccuraciesDict(experiment_id, dataset_name, explanation_names,results_dir="results"):
    accuracies_dict = {}
    for explanation_name in explanation_names:
        accuracies_dict[explanation_name] = []

        accuracies_csv_name = experiment_id+"_"+dataset_name+"_"+explanation_name+"_results.csv"
        accuracies_csv_path = os.path.join(results_dir,accuracies_csv_name)

        accuracies_string = ""

        with open(accuracies_csv_path, "r") as f:
            accuracies_string = f.read()
        
        accuracies_lines = accuracies_string.split("\n")
        
        for accuracy_line in accuracies_lines:
            accuracies = accuracy_line.split(",")
            accuracies_dict[explanation_name].append(float(accuracies[2]))
    
    return accuracies_dict


def GetResultsDict(experiment_id, dataset_name, explanation_names,results_dir="results"):
    results_pickle_name = "results_"+experiment_id+"_"+dataset_name+".pkl"
    results_pickle_path = os.path.join(results_dir,results_pickle_name)

    image_results_dicts = {}

    if(os.path.exists(results_pickle_path)):
        print("Results Pickle Found - Loading Results Dict")
        with open(results_pickle_path,"rb") as f:
            image_results_dicts = pickle.load(f)    
    else:    
        print("Results Pickle Not Found - Generating Results Dict")
        
        for explanation_name in explanation_names[:]:
            image_results_dicts[explanation_name] = []

            for deterioration_step in range(20)[:]: 
                deterioration_step_string = format(deterioration_step, '05d')

                file_name = experiment_id + "_" + dataset_name + "_" + explanation_name + "_" + deterioration_step_string +"_deterioration_results.csv"
                file_path = os.path.join(results_dir,"image_results",file_name)

                results_string = ""

                with open(file_path, "r") as f:
                    results_string = f.read()
                
                results = results_string.split("\n")
                
                for img_i in range(len(results))[:]:
                    result = results[img_i].split(",")

                    if(deterioration_step == 0):
                        image_results_dicts[explanation_name].append({"img_i":img_i,"ground_truth":int(result[11]),"original_prediction":int(result[10]),"results":[]})
                        image_results_dicts[explanation_name][img_i]["original_prediction_score"] = float(result[image_results_dicts[explanation_name][-1]["original_prediction"]])
                        image_results_dicts[explanation_name][img_i]["original_prediction_degradations"] = []
                    
                    original_score = image_results_dicts[explanation_name][img_i]["original_prediction_score"]
                    original_prediction = image_results_dicts[explanation_name][img_i]["original_prediction"]
                    current_score = float(result[original_prediction])

                    image_results_dicts[explanation_name][img_i]["original_prediction_degradations"].append(current_score/original_score)
                    image_results_dicts[explanation_name][img_i]["results"].append([float(p)for p in result[:10]])
        
        print("Saving Results Dict as Pickle")
        with open(results_pickle_path,"wb") as f:
            pickle.dump(image_results_dicts, f)    


    return image_results_dicts



def AggregatePredictionStrengths(explanation_names, image_results_dict):
    aggregated_prediction_strengths = {}


if __name__ == "__main__":
    experiment_id = "testROAR_PRESERVATION"
    dataset_name = "CIFAR-10"

    explanation_names = [
        "LIME"
        # ,"Shap"
        # ,"random"
        ]

    image_results_dict = GetResultsDict(experiment_id,dataset_name,explanation_names)

    accuracies_dict = GetAccuraciesDict(experiment_id,dataset_name,explanation_names)

    explanation_name = explanation_names[0]
    image_i = 0

    DisplayPredictionStrengthsAcrossAllClassesForOneExplanationOneImage(explanation_name, image_i, image_results_dict)

    DisplayPredictionStrengthOfPredicitedClassForAllExplanationOneImage(explanation_names, image_i, image_results_dict)
    
    DisplayTestAccuraciesForAllClassesForAllExplanationsAcrossAllImages(explanation_names,accuracies_dict)

    DisplayOriginalPredictionDegradationForAllExplanationsForOneImage(explanation_names, image_i, image_results_dict)

    DisplayOriginalPredictionDegradationForAllExplanationsAcrossAllImages(explanation_names, image_results_dict)