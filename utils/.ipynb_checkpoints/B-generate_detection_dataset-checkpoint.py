from yolo_grpc_client import YoloClient

import glob
import pandas as pd

import os

cams = ('jervskogen_1', 'jervskogen_2', 'nilsbyen_2', 'nilsbyen_3', 'skistua', 'ronningen_1')
share_folder_path = '../data/images/ie/idi/norwai/svv/ski-trd/'


def main():
    # Communication with grpc service is externally handled by the client class
    client = YoloClient(target='ai4eu.idi.ntnu.no:8055')

    df = dict()
    for cam in cams:
        # No need to iterate over all images, if we already have dataset. Check up until when we have data, and start from there
        df_prev = pd.read_csv('../data/datasets/'+cam+'.csv')

        last_timestamp = pd.Timestamp(df_prev.iloc[-1]['timestamp'])
        
        # Finds all filenames starting with the camera name
        imgs = glob.glob(share_folder_path + cam + '*')

        # First, store all detections in a list, and then create a dataframe out of it
        data = []
        for img in imgs:
            try:
                # Get timestamp from img name. TODO: move out of the loop. All these detections come from the same image, so timestamp is the same
                # splits path into folders and file and keep the last one
                # E.g.: ('../images/ie/idi/norwai/svv/ski-trd/jervskogen_1_2021-12-11_09-00-03.png') -> jervskogen_1_2021-12-11_09-00-03.png'
                _, stamp = os.path.split(img)
                
                # Remove camera name
                # E.g. 'jervskogen_1_2021-12-11_09-00-03.png' -> '2021-12-11_09-00-03.png'
                stamp = stamp.replace(cam + '_', '')
                
                # Remove file extension
                # E.g. '2021-12-11_09-00-03.png' -> '2021-12-11_09-00-03'
                stamp = stamp.replace('.png', '')
                
                # Split between date and time
                # E.g. '2021-12-11_09-00-03' -> '2021-12-11' + '09-00-03'
                date, time = stamp.split('_')
                
                # Replace hiffen in time with colon
                # E.g. '09-00-03' -> '09:00:03'
                time = time.replace('-', ':')
            
                # Finally, build the final timestamp string
                # E.g. '2021-12-11' + 'T' + '09:00:03' -> '2021-12-11T09:00:03'
                stamp = date + 'T' + time
                
                stamp = pd.Timestamp(stamp)
                
                if stamp > last_timestamp:
                    detected_objects = client.call_detection_service(img)

                    if len(detected_objects.objects) > 0:
                        for obj in detected_objects.objects:
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

        df_new = pd.concat([df_prev,pd.DataFrame(data,
                                                 columns=['timestamp', 'p1x', 'p1y', 'p2x', 'p2y', 'conf', 'class'])])

        df_new = df_new.sort_values(by='timestamp')

        if cam in ['jervskogen_1', 'jervskogen_2', 'ronningen_1', 'skistua']:
            df_new = df_new[(df_new['timestamp'] < '2022-02-16 09:40:03') | (df_new['timestamp'] > '2022-02-21 19:10:03')]
            df_new = df_new[(df_new['timestamp'] < '2022-02-14 15:30:04') | (df_new['timestamp'] > '2022-02-16 09:30:03')]

        df_new.to_csv('../data/datasets/' + cam + '.csv', index=False)


if __name__ == '__main__':
    main()
