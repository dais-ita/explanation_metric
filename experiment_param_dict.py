import os
# THIS IS NOT FINISHED YET!!!!!
param_dict = {
    "0": { # Shap&LIME, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "0": { # Shap&LIME, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_LIME_1559235451.pkl"),
            "Shap": os.path.join("pixel_lists","deletion_game_CIFAR-10_mean_Shap_1559300187.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },










    "1": { # randomBaseline-123, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 123 
    },
    "2": { # randomBaseline-234, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 234
    },
    "3": { # randomBaseline-345, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 345
    },
    "4": { # randomBaseline-456, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 456
    },
    "5": { # randomBaseline-567, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 567
    },
    "6": { # randomBaseline-678, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 678
    },
    "7": { # randomBaseline-789, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 789
    },
    "8": { # randomBaseline-890, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 890
    },
    "9": { # randomBaseline-901, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 901
    },
    "10": { # randomBaseline-120, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 120
    },
    "11": { # randomBaseline-123, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # True
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 123 
    },
    "12": { # randomBaseline-234, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 234
    },
    "13": { # randomBaseline-345, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 345
    },
    "14": { # randomBaseline-456, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 456
    },
    "15": { # randomBaseline-567, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 567
    },
    "16": { # randomBaseline-678, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 678
    },
    "17": { # randomBaseline-789, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 789
    },
    "18": { # randomBaseline-890, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 890
    },
    "19": { # randomBaseline-901, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 901
    },
    "20": { # randomBaseline-120, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 120
    },
    "21": { # randomBaseline-123, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 123 
    },
    "22": { # randomBaseline-234, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 234
    },
    "23": { # randomBaseline-345, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 345
    },
    "24": { # randomBaseline-456, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 456
    },
    "25": { # randomBaseline-567, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 567
    },
    "26": { # randomBaseline-678, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 678
    },
    "27": { # randomBaseline-789, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 789
    },
    "28": { # randomBaseline-890, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 890
    },
    "29": { # randomBaseline-901, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 901
    },
    "30": { # randomBaseline-120, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 120
    },
    "31": { # randomBaseline-123, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False, # True
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 123 
    },
    "32": { # randomBaseline-234, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 234
    },
    "33": { # randomBaseline-345, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 345
    },
    "34": { # randomBaseline-456, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 456
    },
    "35": { # randomBaseline-567, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 567
    },
    "36": { # randomBaseline-678, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 678
    },
    "37": { # randomBaseline-789, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 789
    },
    "38": { # randomBaseline-890, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 890
    },
    "39": { # randomBaseline-901, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 901
    },
    "40": { # randomBaseline-120, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": 120
    },









    "1": { # randomBaseline-123, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 123 
    },
    "2": { # randomBaseline-234, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "3": { # randomBaseline-345, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "4": { # randomBaseline-456, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "5": { # randomBaseline-567, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "6": { # randomBaseline-678, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "7": { # randomBaseline-789, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "8": { # randomBaseline-890, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "9": { # randomBaseline-901, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "10": { # randomBaseline-120, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
    "11": { # randomBaseline-123, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # True
        "explicit_pixels_per_step": 1,
        "random_seed": 123 
    },
    "12": { # randomBaseline-234, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "13": { # randomBaseline-345, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "14": { # randomBaseline-456, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "15": { # randomBaseline-567, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "16": { # randomBaseline-678, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "17": { # randomBaseline-789, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "18": { # randomBaseline-890, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "19": { # randomBaseline-901, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "20": { # randomBaseline-120, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
    "21": { # randomBaseline-123, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 123 
    },
    "22": { # randomBaseline-234, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "23": { # randomBaseline-345, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "24": { # randomBaseline-456, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "25": { # randomBaseline-567, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "26": { # randomBaseline-678, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "27": { # randomBaseline-789, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "28": { # randomBaseline-890, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "29": { # randomBaseline-901, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "30": { # randomBaseline-120, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
    "31": { # randomBaseline-123, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False, # True
        "explicit_pixels_per_step": 1,
        "random_seed": 123 
    },
    "32": { # randomBaseline-234, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "33": { # randomBaseline-345, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "34": { # randomBaseline-456, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "35": { # randomBaseline-567, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "36": { # randomBaseline-678, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "37": { # randomBaseline-789, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "38": { # randomBaseline-890, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "39": { # randomBaseline-901, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "40": { # randomBaseline-120, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
}