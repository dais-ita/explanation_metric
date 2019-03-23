import os

#visualisations
# X=detrioration_step, y=  one explanation technique - one image - 10 rates of prediction stength for each class (mark the correct class and the original prediction)
# X=detrioration_step, y=  each explanation technique - one image - one percentage of prediction stength for original class
# X=detrioration_step, y=  each explanation technique - aggreagte images - one percentage of prediction stength for original class
# X=detrioration_step, y=  each explanation technique - aggreagte images - one testset accuracy



#"testRETEST_CIFAR-10_LIME_00000_deteriation_results.csv"
experiment_id = "testRETEST"
dataset_name = "CIFAR-10"
explanation_name = "LIME"

for deterioration_step in range(20): 
    deterioration_step_string = format(deterioration_step, '05d')

    file_path = experiment_id + "_" + dataset_name + "_" + explanation_name + "_" + deterioration_step_string + ".csv"