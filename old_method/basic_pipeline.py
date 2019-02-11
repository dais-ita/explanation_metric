import numpy as np

import requests
import json

import cv2

import base64
from PIL import Image
from StringIO import StringIO


def GetDatasetDetails(dataset_name, api_base_url = "http://localhost:3100"):
    dataset_details_url_path = "/dataset-details?type=json&dataset={dataset_name}" 
    dataset_details_url = api_base_url + dataset_details_url_path.format(dataset_name=dataset_name)
    return requests.get(dataset_details_url).json()

    
def GetModelDetails(model_name, api_base_url = "http://localhost:3100"):
    model_details_url_path = "/model-details?type=json&model={model_name}" 
    model_details_url = api_base_url + model_details_url_path.format(model_name=model_name)
    return requests.get(model_details_url).json()
   
   
def GetExplanationDetails(explanation_name, api_base_url = "http://localhost:3100"):
    explanation_details_url_path = "/explanation-details?type=json&explanation={explanation_name}" 
    explanation_details_url = api_base_url + explanation_details_url_path.format(explanation_name=explanation_name)
    return requests.get(explanation_details_url).json()

def DirectExplanationCall(image,dataset_json,model_json,explanation_json, port="6201"):
    base_explain_api = "http://localhost:{port}".format(port=port)
    explain_url_path = "/explanations/explain"
    explain_url = base_explain_api + explain_url_path
    if(type(image) != "str"):
        input_image = encIMG64(image)
    else: 
        input_image = image
    
    post_data = {"input":input_image,"selected_dataset_json":dataset_json,"selected_model_json":model_json,"selected_explanation_json":explanation_json}
    
    return requests.post(explain_url, data=json.dumps(post_data)).json()
    

def readb64(base64_string,convert_colour=True):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(base64_string))
    pimg = Image.open(sbuf)
    if(convert_colour):
    	return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    else:
    	return np.array(pimg) 

def encIMG64(image,convert_colour = False):
    if(convert_colour):
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    retval, img_buf = cv2.imencode('.jpg', image)
    
    return base64.b64encode(img_buf)
     
def CreateOrderedPixelsList(attribution_map):
    pixel_weight_list = []
    for i in range(attribution_map.shape[0]):
        for j in range(attribution_map.shape[1]):
            #TODO: decide best way to aggregate colour attribution
            if(len(attribution_map[i][j].shape) > 0):
                attribution_value = sum(attribution_map[i][j])
            else:
                attribution_value = attribution_map[i][j]
            pixel_weight_list.append( (i,j,attribution_value) )
    #TODO: confirm taking the abs is correct
    return sorted(pixel_weight_list,key=lambda x: abs(x[2]),reverse=True)

#config
api_base_url = "http://localhost:3100"

dataset_name = "Traffic Congestion Image Classification (Resized)"
model_name = "vgg16_imagenet"
explanation_name = "LRP"

image_name = "00140_congested.jpg"


#get image
image_fetch_url_path = "/dataset-test-image?dataset={dataset_name}&image={image_name}"
image_fetch_url = api_base_url + image_fetch_url_path.format(dataset_name=dataset_name,image_name=image_name)
input_image_json = requests.get(image_fetch_url).json()
input_image_string = input_image_json["input"]
input_image = readb64(input_image_string, False)


#get item jsons
dataset_json = GetDatasetDetails(dataset_name)
model_json = GetModelDetails(model_name)
explanation_json = GetExplanationDetails(explanation_name)

#get prediction and explanation
results_json  = DirectExplanationCall(input_image,dataset_json,model_json,explanation_json)

prediction_index = np.argmax(results_json['additional_outputs']["prediction_scores"])
initial_prediction_prob = results_json['additional_outputs']["prediction_scores"][prediction_index]

#use explanation technique to determine important information
attribution_map =  np.array(results_json["additional_outputs"]["attribution_map"])
pixel_weight_list = CreateOrderedPixelsList(attribution_map)
        
#deteriorate contributors
deteriation_results = []
deteriation_results.append( initial_prediction_prob )
    
deteriation_rate = 0.02
deteriation_index_step = int(len(pixel_weight_list) * deteriation_rate)

working_image = np.copy(input_image)
for deteriation_step in range(int(1 / deteriation_rate)):
    deteriation_pixels = pixel_weight_list[deteriation_step*deteriation_index_step:min(len(pixel_weight_list)-1, (deteriation_step+1)*deteriation_index_step)]
    
    for px in deteriation_pixels:
        working_image[px[0]][px[1]] = [0,0,0]
    
    #get new prediction
    explanation_result = DirectExplanationCall(working_image,dataset_json,model_json,explanation_json)
    
    prediction_prob = explanation_result["additional_outputs"]["prediction_scores"][prediction_index]
    
    deteriation_results.append( prediction_prob )
        
#compare results
for result in deteriation_results:
    print(result)        
    