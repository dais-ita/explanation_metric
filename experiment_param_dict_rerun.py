import os

param_dict = {
    "0": { # Shap&LIME, mean_perturb, preservation, percentPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME","InputTimesGradient","DeepTaylor","Gradient"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl"),
            "Gradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_Gradient_1563289738.pkl")
        },
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
        "explanation_names": ["Shap","LIME","InputTimesGradient","DeepTaylor","Gradient"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl"),
            "Gradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_Gradient_1563289738.pkl")
        },
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
        "explanation_names": ["Shap","LIME","InputTimesGradient","DeepTaylor","Gradient"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl"),
            "Gradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_Gradient_1563289738.pkl")
        },
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
        "explanation_names": ["Shap","LIME","InputTimesGradient","DeepTaylor","Gradient"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl"),
            "Gradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_Gradient_1563289738.pkl")
        },
        "perturb_method": "random",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None # 123, 234, 345, 456, 567, 678, 789, 890, 901, 120
    },
    "4": { # Shap&LIME, grid_perturb, preservation, 1pxPerStep
        "dataset_name": "CIFAR-10-original", # always the same
        "model_name": "vgg16_richard",       # always the same
        "normalise_data": True,              # always the same
        "load_base_model_if_exist": True,    # always the same
        "save_pixel_list": False,            # always the same
        "deterioration_rate": 0.05,          # always the same
        "explanation_names": ["Shap","LIME","InputTimesGradient","DeepTaylor","Gradient"], # random
        "load_from_pixel_list_path_dict": {
            "LIME": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_LIME_1562778871.pkl"),
            "Shap": os.path.join("pixel_lists","richard_preservation_game_CIFAR-10-original_mean_Shap_1562776643.pkl"),
            "InputTimesGradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_InputTimesGradient_1563244779.pkl"),
            "DeepTaylor": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_DeepTaylor_1563260594.pkl"),
            "Gradient": os.path.join("pixel_lists", "richard_1px_deletion_game_CIFAR-10-original_mean_Gradient_1563289738.pkl")
        },
        "perturb_method": "grid",
        "experiment_id": "richard_",
        "use_deletion_game": False,
        "explicit_pixels_per_step": 1,
        "random_seed": None
    },
    "5": { # randomBaseline-123, mean_perturb, preservation, percentPerStep
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
    "6": { # randomBaseline-234, mean_perturb, preservation, percentPerStep
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
    "7": { # randomBaseline-345, mean_perturb, preservation, percentPerStep
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
    "8": { # randomBaseline-456, mean_perturb, preservation, percentPerStep
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
    "9": { # randomBaseline-567, mean_perturb, preservation, percentPerStep
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
    "10": { # randomBaseline-123, random_perturb, preservation, percentPerStep
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
    "11": { # randomBaseline-234, random_perturb, preservation, percentPerStep
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
    "12": { # randomBaseline-345, random_perturb, preservation, percentPerStep
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
    "13": { # randomBaseline-456, random_perturb, preservation, percentPerStep
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
    "14": { # randomBaseline-567, random_perturb, preservation, percentPerStep
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
    "15": { # randomBaseline-123, mean_perturb, preservation, 1pxPerStep
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
    "16": { # randomBaseline-234, mean_perturb, preservation, 1pxPerStep
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
    "17": { # randomBaseline-345, mean_perturb, preservation, 1pxPerStep
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
    "18": { # randomBaseline-456, mean_perturb, preservation, 1pxPerStep
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
    "19": { # randomBaseline-567, mean_perturb, preservation, 1pxPerStep
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
    "20": { # randomBaseline-123, random_perturb, preservation, 1pxPerStep
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
    "21": { # randomBaseline-234, random_perturb, preservation, 1pxPerStep
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
    "22": { # randomBaseline-345, random_perturb, preservation, 1pxPerStep
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
    "23": { # randomBaseline-456, random_perturb, preservation, 1pxPerStep
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
    "24": { # randomBaseline-567, random_perturb, preservation, 1pxPerStep
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
    "25": { # randomBaseline-678, mean_perturb, preservation, percentPerStep
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
    "26": { # randomBaseline-789, mean_perturb, preservation, percentPerStep
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
    "27": { # randomBaseline-890, mean_perturb, preservation, percentPerStep
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
    "28": { # randomBaseline-901, mean_perturb, preservation, percentPerStep
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
    "29": { # randomBaseline-120, mean_perturb, preservation, percentPerStep
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
    "30": { # randomBaseline-678, random_perturb, preservation, percentPerStep
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
    "31": { # randomBaseline-789, random_perturb, preservation, percentPerStep
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
    "32": { # randomBaseline-890, random_perturb, preservation, percentPerStep
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
    "33": { # randomBaseline-901, random_perturb, preservation, percentPerStep
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
    "34": { # randomBaseline-120, random_perturb, preservation, percentPerStep
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
    "35": { # randomBaseline-678, mean_perturb, preservation, 1pxPerStep
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
    "36": { # randomBaseline-789, mean_perturb, preservation, 1pxPerStep
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
    "37": { # randomBaseline-890, mean_perturb, preservation, 1pxPerStep
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
    "38": { # randomBaseline-901, mean_perturb, preservation, 1pxPerStep
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
    "39": { # randomBaseline-120, mean_perturb, preservation, 1pxPerStep
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
    "40": { # randomBaseline-678, random_perturb, preservation, 1pxPerStep
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
    "41": { # randomBaseline-789, random_perturb, preservation, 1pxPerStep
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
    "42": { # randomBaseline-890, random_perturb, preservation, 1pxPerStep
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
    "43": { # randomBaseline-901, random_perturb, preservation, 1pxPerStep
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
    "44": { # randomBaseline-120, random_perturb, preservation, 1pxPerStep
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
    "45": { # randomBaseline-123, grid_perturb, preservation, 1pxPerStep
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
    "46": { # randomBaseline-789, grid_perturb, preservation, 1pxPerStep
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
    "47": { # randomBaseline-890, grid_perturb, preservation, 1pxPerStep
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
    "48": { # randomBaseline-901, grid_perturb, preservation, 1pxPerStep
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
    "49": { # randomBaseline-120, grid_perturb, preservation, 1pxPerStep
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
    "50": { # randomBaseline-678, grid_perturb, preservation, 1pxPerStep
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
    "51": { # randomBaseline-789, grid_perturb, preservation, 1pxPerStep
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
    "52": { # randomBaseline-890, grid_perturb, preservation, 1pxPerStep
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
    "53": { # randomBaseline-901, grid_perturb, preservation, 1pxPerStep
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
    "54": { # randomBaseline-120, grid_perturb, preservation, 1pxPerStep
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
    }
}
