import os

param_dict = {
    "0": { # Shap&LIME, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": True,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "1": { # Shap&LIME, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "2": { # Shap&LIME, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "3": { # Shap&LIME, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "4": { # Shap&LIME, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": True,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "5": { # Shap&LIME, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "6": { # Shap&LIME, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "7": { # Shap&LIME, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["EdgeDetection"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    }
}