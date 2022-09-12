#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pipeline
# Sensor -> Capture Image -> Policy -> Transmit Image
# Cloud -> Send heatmap to sensor 


# In[2]:


#%load_ext autoreload
#%autoreload 2


# In[47]:


import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import image_slicer 
from PIL import Image, ImageFile
from icecream import ic
import glob
from math import sqrt, ceil, floor
from tqdm.notebook import trange, tqdm
import io                      # for image2byte array
from functions import image_to_byte_array, nlargest_indices, adjust_indices_for_slicerPackage, select_tiles, make_timestamp
from functions import overlap_selected_tiles_on_background_image
import os


## Todo: Cloud part needs to be updated
import grpc                     # for cloud inference

from image_slicer import slice
import logging
import warnings
import fnmatch

_YOLO_PORT = 8055

#yolo_channel = grpc.insecure_channel("ai4eu.idi.ntnu.no:" + str(_YOLO_PORT))
#yolo_stub = yolov5_service_pb2_grpc.YoloV5Stub(yolo_channel)
'''
YOLOv5 🚀 2022-5-21 Python-3.7.11 torch-1.10.2 CPU

Fusing layers... 
[W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.5 GFLOPs
Adding AutoShape... 
'''


# In[35]:


def update_heatmap_alpha(distribution_heatmap, bins_x, bins_y, detected_objects, confidence_threshold: float, alpha: float):

    # Image parameters
    image_num_grid_tiles = 64   # This should be an even number, total tilles to be in image 
    grid_width = int(ceil(sqrt(image_num_grid_tiles))) # number of columns
    grid_height = int(ceil(image_num_grid_tiles / float(grid_width))) # number of rows  
    
    # Initialise heatmap 
    distribution_heatmap_local = np.zeros([grid_height, grid_width])
    
    if len(detected_objects) > 0:

        for index, row in detected_objects.iterrows():

            if row['name'] == 'person' and row['confidence'] > confidence_threshold:
                # Top left point
                p1 = [row['xmin'], row['ymin']]

                pd1 = [np.digitize(p1[0],bins_x) - 1, np.digitize(p1[1],bins_y) - 1]

                # Bottom right point
                p2 = [row['xmax'], row['ymax']]

                pd2 = [np.digitize(p2[0],bins_x) - 1, np.digitize(p2[1],bins_y) - 1]

                # Increment heatmap matrix
                distribution_heatmap_local[np.ix_(np.arange(pd1[1],pd2[1]+1), np.arange(pd1[0],pd2[0]+1))] += 1

    for i in range(len(distribution_heatmap_local)):
        for j in range(len(distribution_heatmap_local[i])): 
                distribution_heatmap[i,j] = ((1-alpha) * distribution_heatmap[i,j]) + ((alpha)*distribution_heatmap_local[i,j])
    
    return distribution_heatmap         

    '''
    if NORMALIZE_HEATMAPS:
        detect_heatmap_partition = detect_heatmap_partition/detect_heatmap_partition.sum()

    ax = sns.heatmap(detect_heatmap_partition, xticklabels=False, yticklabels=False, cmap='Reds')
    plt.title(cam)

    if SAVE_IMAGES:
        plt.savefig('outputs/' + 'partition_heatmap_' + cam + '.jpg', dpi=300, format='jpg')

    plt.clf()
    '''    


# In[36]:


def get_camera_background_image_path(camera_name:str, time):
   
    folder_path = "../utils/reference-images/"
    
    if ((time.hour >= 18) or (time.hour <= 10)):
        # Night
        cam_background_images_path = {
            "jervskogen_1": folder_path + "jervskogen_1_2021-12-11_11-30-03.png",
            "jervskogen_2": folder_path + "jervskogen_2_2021-12-17_03-30-04.png",
            "nilsbyen_2":   folder_path + "nilsbyen_2_2021-12-11_11-10-03.png",
            "nilsbyen_3":   folder_path + "nilsbyen_3_2021-12-11_10-00-03.png",
            "skistua":      folder_path + "skistua_2021-12-11_10-00-03.png",
            "ronningen_1":  folder_path + "jervskogen_1_2021-12-11_11-30-03.png"    
        }
        
    else:   
        # Day
        cam_background_images_path = {
            "jervskogen_1": folder_path + "jervskogen_1_2021-12-11_11-30-03.png",
            "jervskogen_2": folder_path + "jervskogen_2_2021-12-11_09-50-03.png",
            "nilsbyen_2":   folder_path + "nilsbyen_2_2021-12-11_11-10-03.png",
            "nilsbyen_3":   folder_path + "nilsbyen_3_2021-12-11_10-00-03.png",
            "skistua":      folder_path + "skistua_2021-12-11_10-00-03.png",
            "ronningen_1":  folder_path + "jervskogen_1_2021-12-11_11-30-03.png"    
        }
   
    return cam_background_images_path[camera_name]


# In[37]:


def n_percentage_of_y(percentage: int, whole: int):
    
    return round ((percentage * whole) / 100.0)


# In[38]:
def draw_tiles_for_exploitation_phase(all_tiles_1d_indices, number_of_tiles_to_select, heatmap_distribution_1d):
    
    
    # Here heatmap_distribution_1d as input should not be normalized.
    # number_of_tiles_to_select: Translates to transmisison level. OR The number of tiles we want to draw
    # all_tiles_1d_indices: The indices represent tiles
    
    heatmap_distribution_1d = np.array(heatmap_distribution_1d)
   
    if np.sum(heatmap_distribution_1d) == 0:
        # Draw based on uniform distribution
        tiles_to_select = np.random.choice(all_tiles_1d_indices, size = number_of_tiles_to_select, replace=False)
    else:
        
        all_tiles_1d_nonzero_elements_indices = np.nonzero(heatmap_distribution_1d)[0] # The non-zero func returns tuple that is why we add "[0]" in the end to access np array only 
        heatmap_distribution_1d_normalize = heatmap_distribution_1d/np.sum(heatmap_distribution_1d) # Normalize
        
        
        if all_tiles_1d_nonzero_elements_indices.size >= number_of_tiles_to_select:
            # Draw tiles based on distribution (General Case for exploitation phase)
            tiles_to_select = np.random.choice(all_tiles_1d_indices, size = number_of_tiles_to_select, replace=False, p = heatmap_distribution_1d_normalize)
        else:
            # Exception case
            if all_tiles_1d_nonzero_elements_indices.size < number_of_tiles_to_select:
                # First draw tiles contaning the non-zero value, here the size is based on non-zero values
                tiles_to_select = np.random.choice(all_tiles_1d_indices, size =all_tiles_1d_nonzero_elements_indices.size, replace=False, p = heatmap_distribution_1d_normalize)
                
                # Check how many more we have to draw based on the transmission level i.e. number_of_tiles_to_select
                number_of_tiles_to_draw_remaining = number_of_tiles_to_select - all_tiles_1d_nonzero_elements_indices.size
                all_tiles_1d_Zero_elements_indices =  np.where(heatmap_distribution_1d == 0)[0]
                
                # Draw remaining tiles having zero value based on uniform distribution
                tiles_to_select = np.append(tiles_to_select, np.random.choice(all_tiles_1d_Zero_elements_indices, size =number_of_tiles_to_draw_remaining, replace=False))

    return tiles_to_select  

def egreedy_policy(number_of_tiles_to_select: int, total_image_tiles: int, grid_width, grid_height, e_decay: float, heatmap_distribution_2d):
    
    all_tiles_1d = np.arange(total_image_tiles) # Creates a 1-D array from 0-n, these numbers represent tiles number
    
    p = np.random.random()
    exploration_bool = True
    if p < e_decay:
        # Exploration
        uniform_distribution = [1/total_image_tiles] * total_image_tiles # create uniform distribution 1D array
        tiles_to_select = np.random.choice(all_tiles_1d, size = number_of_tiles_to_select, replace=False, p = uniform_distribution)
    
    else:
        # Exploitation
        #heatmap_distribution_2d = heatmap_distribution_2d/heatmap_distribution_2d.sum() # Normalize, for old** 
        heatmap_distribution_1d = list(np.concatenate(heatmap_distribution_2d).flat)    # convert to 1D array
        #tiles_to_select = np.random.choice(all_tiles_1d, size = number_of_tiles_to_select, replace=False, p = heatmap_distribution_1d) # Old**
        # The following is new draw function and normalizes inside the function
        tiles_to_select = draw_tiles_for_exploitation_phase(all_tiles_1d, number_of_tiles_to_select, heatmap_distribution_1d)

        exploration_bool = False



    # reshape to 2D array and find indices of selected tiles
    all_tiles_number_2d = np.reshape(all_tiles_1d, (grid_width, grid_height)) # (rows, columns)
    boolean_2D_array = np.isin(all_tiles_number_2d, tiles_to_select)              # Returns a boolean 2D array, where tiles to select are marked True
    indices_x, indices_y = np.where(boolean_2D_array == True)                     # Returns indices of tiles marked as true
    
    return {
            "selected_tiles_inidces_x": indices_x,
            "selected_tiles_indices_y": indices_y,
            "if_exploration": exploration_bool
            }
        


# In[39]:


def exploitation_always(number_of_tiles_to_select: int, total_image_tiles: int, grid_width, grid_height, heatmap_distribution_2d, image_tiles, cam: str, EXP_CONF_THRESHOLD, model, timestamp):
        all_tiles_1d = np.arange(total_image_tiles) # Creates a 1-D array from 0-n, these numbers represent tiles number

         # Exploitation
        heatmap_distribution_2d = heatmap_distribution_2d/heatmap_distribution_2d.sum() # Normalize
        heatmap_distribution_1d = list(np.concatenate(heatmap_distribution_2d).flat)    # convert to 1D array
        tiles_to_select = np.random.choice(all_tiles_1d, size = number_of_tiles_to_select, replace=False, p = heatmap_distribution_1d)
    
        # reshape to 2D array and find indices of selected tiles
        all_tiles_number_2d = np.reshape(all_tiles_1d, (grid_width, grid_height)) # (rows, columns)
        boolean_2D_array = np.isin(all_tiles_number_2d, tiles_to_select)              # Returns a boolean 2D array, where tiles to select are marked True
        indices_x, indices_y = np.where(boolean_2D_array == True)                     # Returns indices of tiles marked as true
        selected_tiles_inidces_x = indices_x
        selected_tiles_indices_y = indices_y
        
        # Map selected tiles indices as per SlicerPackage indices scheme 
        selected_tiles_indices =  adjust_indices_for_slicerPackage(selected_tiles_inidces_x,
                                                                           selected_tiles_indices_y)  
                
        # Select only chosen image tiles from all image tiles based on policy
        selected_tiles = select_tiles(image_tiles, selected_tiles_indices)

        # Paste selected tiles on reference/background image                
        overlapped_image = overlap_selected_tiles_on_background_image(tiles_to_overlap = selected_tiles,
                                                                              total_number_of_tiles = total_image_tiles,
                                                                              reference_image_path = get_camera_background_image_path(cam, timestamp)
                                                                              )
        # Inference
        # (A) Perform inference on machine                    
        results = model(overlapped_image)    
        detected_objects = results.pandas().xyxyn[0]

        if len(detected_objects) > 0:
            person_count = 0
                        
            for index, row in detected_objects.iterrows():

                if( (row['name'] == 'person') & (row['confidence'] > EXP_CONF_THRESHOLD)):
                    person_count = person_count + 1

        else:
            person_count = 0
            
        return  {
                "person_count": person_count
                }    

            


# In[40]:


def make_dir_if_not_exists(dir_path):
    # Check whether the specified path exists or not
    is_exist = os.path.exists(dir_path)

    if not is_exist:
      # Create a new directory because it does not exist 
      os.makedirs(dir_path)


# In[45]:


def pipeline_sensor(experiment_parameter):
    
    
    SHARED_DRIVE = False
    INFERENCE_LOCAL = True
    # Set cam and policy_percentage 
    cam, policy_percentage = experiment_parameter
    #ic(experiment_parameter)
    #cams = ('jervskogen_1', 'jervskogen_2', 'nilsbyen_2', 'nilsbyen_3', 'skistua', 'ronningen_1')
    #cam = 'jervskogen_2'
    
    # Execute following line on cmd (cmd location -> inside project folder) of mac to mount the shared network folder on your machine.
    # User is your university's account username
    ## mount_smbfs //user@forskning.it.ntnu.no/nfs/ie/idi/norwai/svv/ski-trd ./Volumes/Ski
    ## After use unmount ->  umount /Volumes/Ski
    if SHARED_DRIVE is True:
        share_folder_path = './Volumes/Ski/'
    else:
        share_folder_path = './local_data/Ski/'

    # Few images are missing some bytes, therefore, we set turncation true 
    # which fills the missing bytes of image with grey color at the time of image loading.
    ImageFile.LOAD_TRUNCATED_IMAGES = True 
    
    # Parameters/initialisation
    #policy_percentages = [20,40,60,80]  # 20, 40, 60, 80 percentage -> refers to energy budget and translates to how much image portion to transmit    
    #policy_percentages = [policy_percentageA]
    MONTHS = ['_2021-12', '_2022-01', '_2022-02', '_2022-03']     # 2022-01-11       # For all images it could be empty with underscore '_' or for month '_2022-01' 
    number_of_months_to_include = 4
    EXP_CONF_THRESHOLD = 0.5
    SAVE_HEATMAP = True
    
    # e_greedy policy parameters
    exploration = 0.5
    decay = 0.9999
    span = 144 * 30             # 144 Samples per day (10 min interval) x 30 days
    alpha = 2/ (span +1)
    
    
    # Image parameters
    image_num_grid_tiles = 64   # This should be an even number, total tilles to be in image 
    grid_width = int(ceil(sqrt(image_num_grid_tiles))) # number of columns
    grid_height = int(ceil(image_num_grid_tiles / float(grid_width))) # number of rows  
    
    exp_name_folder = 'Experiment_operational_egreedy_with_alpha_' + str(exploration)
    
    # tqdm shows the progress bar
    #for policy_percentage in tqdm(policy_percentages):
    # Initialise heatmap 
    heatmap_distribution_global = np.zeros([grid_height, grid_width])

    # Create bins between 0 and 1 to use normalized detection points for heatmaps
    bins_x = np.arange(0,1,1/grid_width)
    bins_y = np.arange(0,1,1/grid_height)


    number_of_tiles_to_select = n_percentage_of_y(policy_percentage, image_num_grid_tiles)

    # Finds all filenames starting with the camera name
    #images_path = glob.glob(share_folder_path + cam + MONTH_TEST + '*')
    # OR
    images_path_all = fnmatch.filter(os.listdir(share_folder_path), cam + "*")
    images_path = []
    for i in range(number_of_months_to_include):  
        images_path.extend(fnmatch.filter(images_path_all, cam + MONTHS[i] + "*"))

    #images_path = images_path[:10]
    # Add folder path before images name, if you don't use glob.
    images_path = [share_folder_path + image_path for image_path in images_path]
    images_path.sort(key=lambda x: os.path.getmtime(x))
    ic(len(images_path))

    # First, store all detections in a list, and then create a dataframe out of it
    all_timestamps_count_results = []
    only_detections_detailed_results = []
    all_timestamps_detections_detailed_results = []


    # For local inference
    # Model
    if INFERENCE_LOCAL is True:
        #logging.getLogger("utils.general").setLevel(logging.WARNING)
        # Suppress PyTorch warnings
        warnings.filterwarnings('ignore', message='User provided device_type of \'cuda\', but CUDA is not available. Disabling')
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False, device='cpu')  # or yolov5m, yolov5l, yolov5x, custom


    for index, image_path in enumerate(tqdm(images_path)):
        try:

            # Slice image into n parts/tiles 
            image_tiles = image_slicer.slice(image_path, image_num_grid_tiles, save=False) # accepts even number



            # Based on policy, choose which image tiles to send 
            e_decay = exploration * decay
            exploration = e_decay
            #ic(heatmap_distribution_global)
            result_policy = egreedy_policy( number_of_tiles_to_select,
                                            image_num_grid_tiles, 
                                            grid_width, grid_height, 
                                            e_decay, heatmap_distribution_global ) 




            # Map selected tiles indices as per SlicerPackage indices scheme 
            selected_tiles_indices =  adjust_indices_for_slicerPackage(result_policy["selected_tiles_inidces_x"],
                                                                       result_policy["selected_tiles_indices_y"])  

            # Select only chosen image tiles from all image tiles based on policy
            selected_tiles = select_tiles(image_tiles, selected_tiles_indices)

            # Check Image Capture time to select background                       
            stamp = make_timestamp(image_path, cam)

            # Paste selected tiles on reference/background image                
            overlapped_image = overlap_selected_tiles_on_background_image(tiles_to_overlap = selected_tiles,
                                                                          total_number_of_tiles = image_num_grid_tiles,
                                                                          reference_image_path = get_camera_background_image_path(cam, pd.Timestamp(stamp))
                                                                          )
            # Inference
            if INFERENCE_LOCAL is True:

                # (A) Perform inference on machine                    
                results = model(overlapped_image)    
                detected_objects = results.pandas().xyxyn[0]

                # Update heatmap
                heatmap_distribution_global = update_heatmap_alpha(heatmap_distribution_global, bins_x, bins_y, detected_objects, EXP_CONF_THRESHOLD, alpha)

               # ic(type(detected_objects))

                # For evaluation of alpha
                entry_detections = {
                "timestamp": pd.Timestamp(stamp),
                "detections": detected_objects,
                "alpha": alpha,
                "exp-conf": EXP_CONF_THRESHOLD,
                "exp-grid_tiles": image_num_grid_tiles,
                "exp-local_inference": INFERENCE_LOCAL,
                "exp-e_decay": e_decay,    
                }
                all_timestamps_detections_detailed_results.append(entry_detections)


                if len(detected_objects) > 0:

                    #stamp = make_timestamp(image_path, cam)
                    person_count = 0

                    for index, row in detected_objects.iterrows():

                        entry_detections = {
                            "timestamp": pd.Timestamp(stamp),
                            "p1x": row['xmin'],
                            "p1y": row['ymin'],
                            "p2x": row['xmax'],
                            "p2y": row['ymax'],
                            "conf": row['confidence'],
                            "class": row['name'],
                            "detected_person": "True",
                            "EXP-policy-tiles": selected_tiles_indices

                        }

                        only_detections_detailed_results.append(entry_detections)

                        if( (row['name'] == 'person') and (row['confidence'] > EXP_CONF_THRESHOLD)):
                            person_count = person_count + 1


                else:
                    entry_detections = {
                            "timestamp": pd.Timestamp(stamp),
                            "p1x": "-1",
                            "p1y": "-1",
                            "p2x": "-1",
                            "p2y": "-1",
                            "conf":"-1",
                            "class": "None",
                            "detected_person": "False",
                            "EXP-policy-tiles": selected_tiles_indices

                    }

                    only_detections_detailed_results.append(entry_detections)
                    stamp = make_timestamp(image_path, cam)
                    person_count = 0



                # We add this to check what could happen in terms of error if we do exploitation instead of exploration. Ideally exploitation should improve the result.
                # The following if block is not part of the pipeline but is used to get error graphs.
                #if (result_policy["if_exploration"] == True):
                #    forced_exploitation_result = exploitation_always(number_of_tiles_to_select,
                #                                image_num_grid_tiles, 
                #                                grid_width, grid_height, 
                #                                heatmap_distribution_global,
                #                                image_tiles, cam, EXP_CONF_THRESHOLD, model, pd.Timestamp(stamp))    
                #    forced_exploitation_person_count = forced_exploitation_result["person_count"]
                #else:
                    # Here we write the result of policy true exploitation
                    #forced_exploitation_person_count = person_count

                # Store results    
                entry_timestamps = {
                    "timestamp": pd.Timestamp(stamp),
                    "count":   person_count,
                    "exp-conf": EXP_CONF_THRESHOLD,
                    "exp-grid_tiles": image_num_grid_tiles,
                    "exp-local_inference": INFERENCE_LOCAL,
                    "exp-e_decay": e_decay,
                    "exp-policy_is_exploration": result_policy["if_exploration"],
                 #   "count_exploitation_forced": forced_exploitation_person_count,
                    "image_path": image_path

                }

                all_timestamps_count_results.append(entry_timestamps)


            else:    
                # (B) Do inference on cloud and send image... 
                ##request = yolov5_service_pb2.Image(data = image_to_byte_array(overlapped_image))
                ##detected_objects = yolo_stub.detect(request) 
                ##detected_objects = detected_objects.objects

                if len(detected_objects) > 0:
                    stamp = make_timestamp(image_path, cam)


                    for obj in detected_objects:
                        data.append([pd.Timestamp(stamp),
                                     obj.p1.x,
                                     obj.p1.y,
                                     obj.p2.x,
                                     obj.p2.y,
                                     obj.conf,
                                     obj.class_name])

        # Error handling: if there is any error inside the container, just ignore and continue processing next images
        except ValueError as error:
            print('Error inside grpc service:')
            print(error.args[0])
            print(error.args[1])
            print('Continue to next image')
            continue


        except Exception as error:
            print ('Error due to some reason')
            print ('index: ')
            print (index)
            print (image_path)
            continue

    df_all_timestamps_count_results = pd.DataFrame(all_timestamps_count_results)
    df_only_detections_detailed_results = pd.DataFrame(only_detections_detailed_results)
    df_all_timetsamps_detections_detailed_results = pd.DataFrame(all_timestamps_detections_detailed_results)


    path_string_detections = exp_name_folder + '/' + cam + '_only_detections_' + '_policy_' + str(policy_percentage) + '_tiles_' + str(image_num_grid_tiles)
    path_string_timestamps = exp_name_folder + '/' + cam + '_all_timestamps_count_' + '_policy_' + str(policy_percentage) + '_tiles_' + str(image_num_grid_tiles)
    path_string_all_timestamps_detections = exp_name_folder + '/' + cam + '_all_timestamps_detections_' + '_policy_' + str(policy_percentage) + '_tiles_' + str(image_num_grid_tiles)

    path_string_detections +=  "_MONTHS"
    path_string_timestamps +=  "_MONTHS"
    path_string_all_timestamps_detections +=  "_MONTHS" 

    for i in range(number_of_months_to_include):  
        path_string_detections +=  str(MONTHS[i])
        path_string_timestamps +=  str(MONTHS[i])
        path_string_all_timestamps_detections +=  str(MONTHS[i])
    database_path = '../data/datasets/'    

    if INFERENCE_LOCAL is True:
        database_path += 'local/'
        make_dir_if_not_exists(database_path + exp_name_folder)
        df_only_detections_detailed_results.to_csv('../data/datasets/local/' + path_string_detections + '.csv', index=False)
        df_all_timestamps_count_results.to_csv('../data/datasets/local/' + path_string_timestamps + '.csv', index=False)
        df_all_timetsamps_detections_detailed_results.to_csv('../data/datasets/local/' + path_string_all_timestamps_detections + '.csv', index=False)

    else:
        database_path += 'cloud/'
        make_dir_if_not_exists(database_path + exp_name_folder)
        df_only_detections_detailed_results.to_csv('../data/datasets/cloud/' +  path_string_detections + '.csv', index=False)
        df_all_timestamps_count_results.to_csv('../data/datasets/cloud/' + path_string_timestamps + '.csv', index=False)

    if SAVE_HEATMAP:
        make_dir_if_not_exists(database_path + exp_name_folder + '/outputs/')
        heatmap_distribution_global = heatmap_distribution_global 
        # Normalize
        heatmap_distribution_global = heatmap_distribution_global/heatmap_distribution_global.sum()
        ax = sns.heatmap(heatmap_distribution_global, xticklabels=False, yticklabels=False, cmap='Reds')
        plt.title(cam)
        plt.savefig(database_path + exp_name_folder + '/outputs/' + 'heatmap_' + cam + '_policy_' + str(policy_percentage) + '.jpg', dpi=300, format='jpg')
        plt.clf()    

#if __name__ == '__main__':
#    pipeline_sensor()        


# In[46]:



#import multiprocess as mp
import multiprocessing as mp
#import tqdm
#from concurrent.futures import ThreadPoolExecutor



print("Number of cpu : ", mp.cpu_count())
num_cpus = mp.cpu_count()


if __name__ == '__main__':
   

    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass


    cams = ('jervskogen_1', 'jervskogen_2', 'nilsbyen_2', 'nilsbyen_3', 'skistua', 'ronningen_1')
    policy_percentages = [20,40,60,80, 100]  # 20, 40, 60, 80 percentage -> refers to energy budget and translates to how much image portion to transmit    
    
    cam1_pp = [20, 40, 60, 80, 100]
    cam1 = ['jervskogen_1'] * len(cam1_pp)
    cam2_pp = [20, 40,60, 80, 100]
    cam2 = ['jervskogen_2'] * len(cam2_pp)
    cam3 = ['nilsbyen_2'] * len(cam2_pp)
    cam4 = ['nilsbyen_3'] * len(cam2_pp)
    cam5 = ['skistua'] * len(cam2_pp)

    experiments_input = list(zip(cam1, cam1_pp))
    experiments_input.extend(list(zip(cam2, cam2_pp)))
    experiments_input.extend(list(zip(cam3, cam2_pp)))
    experiments_input.extend(list(zip(cam4, cam2_pp)))
    experiments_input.extend(list(zip(cam5, cam2_pp)))

    ic(experiments_input)
    
    with mp.Pool(num_cpus) as p:
        print(p.map(pipeline_sensor, experiments_input))
        #p.join()
    
    # exiting the 'with'-block has stopped the pool
    #print("Now the pool is closed and no longer available")
   

    # Super fast
    #with ThreadPoolExecutor(num_cpus) as p:
    #    p.map(pipeline_sensor, experiments_input) 
   

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")       






