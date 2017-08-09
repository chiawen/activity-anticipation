from __future__ import print_function
import os
import csv
import argparse

import numpy as np

from analyzer import DatasetAnalyzer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./',
            help='Data directory')
    parser.add_argument('--video_dir', type=str, default='videos/',
            help='Video directory')
    parser.add_argument('--labels', type=str, default='handShake,highFive,hug,kiss',
            help='Delimited list of labels')
    parser.add_argument('--output', type=str, default='train_file.csv')
    args = parser.parse_args()

    label_list = args.labels.split(',')
    label_to_no = {label:idx for idx,label in enumerate(label_list)}
    print('Label to number')
    print(label_to_no)

    output_path = os.path.join(args.data_dir, args.output)
    out = open(output_path, 'w')
    writer = csv.writer(out, delimiter=',')
    writer.writerow(['id', 'label'])

    video_path = os.path.join(args.data_dir, args.video_dir)
    
    data = []
    labels = []
    counts = {label:0 for label in label_list}

    
    videos = os.listdir(video_path)
    videos = filter(lambda x: x.endswith('avi'), videos)
    
    # Get dataset info
    TVHI_analyzer = DatasetAnalyzer(args.data_dir, 'TVHI', label_list)
    frame_info = TVHI_analyzer.get_frame_info()

    frame_annotations = TVHI_analyzer.video_frame_dict
    
    for video in videos:
        video_name = video[:-4]
        annotation = frame_annotations[video_name]
        T = annotation[0] - 1
        
        if T >= 0 :
            label_name = video[:-9]
            label_no = label_to_no[label_name]
            data.append(video)
            labels.append(label_no)
    
            counts[label_name] += 1 


    for label_name in label_list:
        count = counts[label_name]

        print('Class {}: {} videos'.format(label_name, count))

    data = np.array(data)
    labels = np.array(labels)

    assert data.shape[0] == labels.shape[0], (
            'data.shape: {} labels.shape: {}'.format(data.shape, labels.shape))

    print('Total number of data: {}'.format(data.shape[0]))
    """
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    """

    for i, video in enumerate(data):
        writer.writerow([video, labels[i]])

    print('Wrote to file: {}'.format(output_path))
    

if __name__ == '__main__':
    main()

