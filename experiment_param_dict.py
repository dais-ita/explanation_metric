import os

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
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "1": { # Shap&LIME, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "2": { # Shap&LIME, random_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "3": { # Shap&LIME, random_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": None, # 1 
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "4": { # Shap&LIME, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "5": { # Shap&LIME, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "6": { # Shap&LIME, random_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "7": { # Shap&LIME, random_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "8": { # randomBaseline-123, mean_perturb, deletion, percentPerStep
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
    "9": { # randomBaseline-234, mean_perturb, deletion, percentPerStep
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
    "10": { # randomBaseline-345, mean_perturb, deletion, percentPerStep
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
    "11": { # randomBaseline-456, mean_perturb, deletion, percentPerStep
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
    "12": { # randomBaseline-567, mean_perturb, deletion, percentPerStep
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
    "13": { # randomBaseline-123, mean_perturb, preservation, percentPerStep
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
    "14": { # randomBaseline-234, mean_perturb, preservation, percentPerStep
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
    "15": { # randomBaseline-345, mean_perturb, preservation, percentPerStep
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
    "16": { # randomBaseline-456, mean_perturb, preservation, percentPerStep
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
    "17": { # randomBaseline-567, mean_perturb, preservation, percentPerStep
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
    "18": { # randomBaseline-123, random_perturb, deletion, percentPerStep
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
    "19": { # randomBaseline-234, random_perturb, deletion, percentPerStep
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
    "20": { # randomBaseline-345, random_perturb, deletion, percentPerStep
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
    "21": { # randomBaseline-456, random_perturb, deletion, percentPerStep
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
    "22": { # randomBaseline-567, random_perturb, deletion, percentPerStep
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
    "23": { # randomBaseline-123, random_perturb, preservation, percentPerStep
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
    "24": { # randomBaseline-234, random_perturb, preservation, percentPerStep
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
    "25": { # randomBaseline-345, random_perturb, preservation, percentPerStep
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
    "26": { # randomBaseline-456, random_perturb, preservation, percentPerStep
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
    "27": { # randomBaseline-567, random_perturb, preservation, percentPerStep
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
    "28": { # randomBaseline-123, mean_perturb, deletion, percentPerStep
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
    "29": { # randomBaseline-234, mean_perturb, deletion, 1pxPerStep
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
    "30": { # randomBaseline-345, mean_perturb, deletion, 1pxPerStep
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
    "31": { # randomBaseline-456, mean_perturb, deletion, 1pxPerStep
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
    "32": { # randomBaseline-567, mean_perturb, deletion, 1pxPerStep
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
    "33": { # randomBaseline-123, mean_perturb, preservation, 1pxPerStep
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
    "34": { # randomBaseline-234, mean_perturb, preservation, 1pxPerStep
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
    "35": { # randomBaseline-345, mean_perturb, preservation, 1pxPerStep
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
    "36": { # randomBaseline-456, mean_perturb, preservation, 1pxPerStep
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
    "37": { # randomBaseline-567, mean_perturb, preservation, 1pxPerStep
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
    "38": { # randomBaseline-123, random_perturb, deletion, 1pxPerStep
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
    "39": { # randomBaseline-234, random_perturb, deletion, 1pxPerStep
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
    "40": { # randomBaseline-345, random_perturb, deletion, 1pxPerStep
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
    "41": { # randomBaseline-456, random_perturb, deletion, 1pxPerStep
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
    "42": { # randomBaseline-567, random_perturb, deletion, 1pxPerStep
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
    "43": { # randomBaseline-123, random_perturb, preservation, 1pxPerStep
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
    "44": { # randomBaseline-234, random_perturb, preservation, 1pxPerStep
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
    "45": { # randomBaseline-345, random_perturb, preservation, 1pxPerStep
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
    "46": { # randomBaseline-456, random_perturb, preservation, 1pxPerStep
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
    "47": { # randomBaseline-567, random_perturb, preservation, 1pxPerStep
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
    "48": { # randomBaseline-678, mean_perturb, deletion, percentPerStep
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
    "49": { # randomBaseline-789, mean_perturb, deletion, percentPerStep
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
    "50": { # randomBaseline-890, mean_perturb, deletion, percentPerStep
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
    "51": { # randomBaseline-901, mean_perturb, deletion, percentPerStep
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
    "52": { # randomBaseline-120, mean_perturb, deletion, percentPerStep
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
    "53": { # randomBaseline-678, mean_perturb, preservation, percentPerStep
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
    "54": { # randomBaseline-789, mean_perturb, preservation, percentPerStep
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
    "55": { # randomBaseline-890, mean_perturb, preservation, percentPerStep
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
    "56": { # randomBaseline-901, mean_perturb, preservation, percentPerStep
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
    "57": { # randomBaseline-120, mean_perturb, preservation, percentPerStep
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
    "58": { # randomBaseline-678, random_perturb, deletion, percentPerStep
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
    "59": { # randomBaseline-789, random_perturb, deletion, percentPerStep
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
    "60": { # randomBaseline-890, random_perturb, deletion, percentPerStep
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
    "61": { # randomBaseline-901, random_perturb, deletion, percentPerStep
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
    "62": { # randomBaseline-120, random_perturb, deletion, percentPerStep
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
    "63": { # randomBaseline-678, random_perturb, preservation, percentPerStep
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
    "64": { # randomBaseline-789, random_perturb, preservation, percentPerStep
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
    "65": { # randomBaseline-890, random_perturb, preservation, percentPerStep
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
    "66": { # randomBaseline-901, random_perturb, preservation, percentPerStep
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
    "67": { # randomBaseline-120, random_perturb, preservation, percentPerStep
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
    "68": { # randomBaseline-678, mean_perturb, deletion, 1pxPerStep
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
    "69": { # randomBaseline-789, mean_perturb, deletion, 1pxPerStep
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
    "70": { # randomBaseline-890, mean_perturb, deletion, 1pxPerStep
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
    "71": { # randomBaseline-901, mean_perturb, deletion, 1pxPerStep
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
    "72": { # randomBaseline-120, mean_perturb, deletion, 1pxPerStep
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
    "73": { # randomBaseline-678, mean_perturb, preservation, 1pxPerStep
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
    "74": { # randomBaseline-789, mean_perturb, preservation, 1pxPerStep
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
    "75": { # randomBaseline-890, mean_perturb, preservation, 1pxPerStep
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
    "76": { # randomBaseline-901, mean_perturb, preservation, 1pxPerStep
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
    "77": { # randomBaseline-120, mean_perturb, preservation, 1pxPerStep
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
    "78": { # randomBaseline-678, random_perturb, deletion, 1pxPerStep
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
    "79": { # randomBaseline-789, random_perturb, deletion, 1pxPerStep
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
    "80": { # randomBaseline-890, random_perturb, deletion, 1pxPerStep
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
    "81": { # randomBaseline-901, random_perturb, deletion, 1pxPerStep
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
    "82": { # randomBaseline-120, random_perturb, deletion, 1pxPerStep
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
    "83": { # randomBaseline-678, random_perturb, preservation, 1pxPerStep
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
    "84": { # randomBaseline-789, random_perturb, preservation, 1pxPerStep
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
    "85": { # randomBaseline-890, random_perturb, preservation, 1pxPerStep
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
    "86": { # randomBaseline-901, random_perturb, preservation, 1pxPerStep
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
    "87": { # randomBaseline-120, random_perturb, preservation, 1pxPerStep
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
    "88": { # randomBaseline-123, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 123
    },
    "89": { # randomBaseline-789, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "90": { # randomBaseline-890, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "91": { # randomBaseline-901, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "92": { # randomBaseline-120, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "93": { # randomBaseline-678, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "94": { # randomBaseline-789, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "95": { # randomBaseline-890, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "96": { # randomBaseline-901, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "97": { # randomBaseline-120, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
    "98": { # randomBaseline-123, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 123
    },
    "99": { # randomBaseline-789, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 234
    },
    "100": { # randomBaseline-890, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 345
    },
    "101": { # randomBaseline-901, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 456
    },
    "102": { # randomBaseline-120, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 567
    },
    "103": { # randomBaseline-678, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 678
    },
    "104": { # randomBaseline-789, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 789
    },
    "105": { # randomBaseline-890, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 890
    },
    "106": { # randomBaseline-901, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 901
    },
    "107": { # randomBaseline-120, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["random"],
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": 120
    },
    "108": { # Shap&LIME, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None
    },
    "109": { # Shap&LIME, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
        },
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None
    },
    "110": { # InputTimesGradient, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": True,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean",
        "experiment_id": "richard_",
        "use_deletion_game": True,
        "explicit_pixels_per_step": 1,
        "random_seed": None
    },
    "111": { # DeepTaylor, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": True,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "112": { # InputTimesGradient&DeepTaylor, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "113": { # InputTimesGradient&DeepTaylor, mean_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "114": { # InputTimesGradient&DeepTaylor, rand_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "random", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "115": { # InputTimesGradient&DeepTaylor, rand_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "random", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "116": { # InputTimesGradient&DeepTaylor, mean_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "117": { # InputTimesGradient&DeepTaylor, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # False
        "explicit_pixels_per_step": None,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "118": { # InputTimesGradient&DeepTaylor, rand_perturb, deletion, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "random", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": None,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "119": { # InputTimesGradient&DeepTaylor, mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "random", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # False
        "explicit_pixels_per_step": None,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "120": { # InputTimesGradient&DeepTaylor, grid_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "grid", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "121": { # InputTimesGradient&DeepTaylor, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["InputTimesGradient","DeepTaylor"], # random
        "load_from_pixel_list_path_dict": {
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl")
        },
        "perturb_method": "grid", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": False, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "122": { # Gradient(sensitivity), mean_perturb, deletion, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": True,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Gradient"], # random
        "load_from_pixel_list_path_dict": {},
        "perturb_method": "mean", # random, grid
        "experiment_id": "richard_",
        "use_deletion_game": True, # False
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    }
}
