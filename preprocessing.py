'''
We extract the all frames before the action from each video
'''
from __future__ import print_function
import os
import cv2
import shutil
import argparse

import skimage
import numpy as np
#import scipy.misc

from analyzer import DatasetAnalyzer
from extractor import Extractor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
            help='Data directory')
    parser.add_argument('--video_dir', type=str, default='videos/',
            help='Video directory')
    parser.add_argument('--feature_dir', type=str, default='features/')
    parser.add_argument('--num_frames', type=str, default=30,
            help='Extract the last n frames')
    parser.add_argument('--checkpoint', type=str, 
            default='./inception_checkpoint/inception_v3.ckpt',
            help='Pretrained model')
    args = parser.parse_args()

    video_path = os.path.join(args.data_dir, args.video_dir)
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)

    feature_dir = os.path.join(args.data_dir, args.feature_dir)
    if not os.path.exists(feature_dir):
        os.mkdir(feature_dir)


    # Get dataset info
    TVHI_analyzer = DatasetAnalyzer(args.data_dir, 'TVHI', 
            ['handShake', 'highFive', 'hug', 'kiss'])

    number_info = TVHI_analyzer.get_number_info()
    frame_info = TVHI_analyzer.get_frame_info()
    print(number_info)
    print(frame_info)

    frame_annotations = TVHI_analyzer.video_frame_dict
    

    # get the model
    model = Extractor(checkpoint=args.checkpoint)

    count = 0
    start = 0
    for video in videos:
        count += 1

        # test one video 
        #if count > 1:
        #    break
        
        print('===============================')
        print('Count: {}'.format(count))
        print('Preprocessing {}'.format(video))

        # Get the feature path of this video
        feature_path = os.path.join(feature_dir, video + '-feat.npy')
        
        # Check if the feature of the video have already existed
        if os.path.isfile(feature_path):
            print('Features already existed')
            print('===============================')
            continue

        # Get the frames for this video
        video_fullpath = os.path.join(video_path, video)
        cap = cv2.VideoCapture(video_fullpath)

        video_name = video[:-4]
        annotation  = frame_annotations[video_name]
        T = annotation[0] - 1

        
        print('T: {}'.format(T))

        """
        length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        print(length)
        print(fps)
        """
        
        if T >= 0: 

            frame_count = 0
            frame_list = []
           
            while frame_count <= T:
                ret, frame = cap.read()
                if ret is False:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(frame)
                #scipy.misc.imsave(video_name + '-' + str(frame_count) + '.jpg', frame)
                frame_count += 1
                
            """ 
            # Only the last n frames are used
            if frame_count > args.num_frames:
                frame_list = frame_list[-args.num_frames:]
                frame_count = args.num_frames
            """
            
            print('Number of frames: {}'.format(frame_count))
            print('Dimensions: {} x {}'.format(frame_list[0].shape[1], frame_list[0].shape[0]))
             
            # Extract features from the frame list
            feature_list = []
            for image in frame_list:
                features = model.extract(image)
                feature_list.append(features)
            feature_list = np.array(feature_list)
            
            # Save extracted features
            np.save(feature_path, feature_list)
            print('Saved features: {}'.format(feature_list.shape))
            
        else:
            print('Number of frames: 0')
            print('Skip this video')
        print('===============================')

    

if __name__=='__main__':

    main()
