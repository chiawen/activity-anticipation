from __future__ import division
from __future__ import print_function

import os

class DatasetAnalyzer(object):
    def __init__(self, path, name, labels):
        self.name = name
        self.path = path
        self.labels = labels
        self.label_dict = {}
        self.label_dir = os.path.join(self.path, 'annotations')
    
    def get_label_dict(self):
        for label in self.labels:
            self.label_dict[label] = set()
        files = os.listdir(self.label_dir)
        for f in files:
            if f.endswith('.annotations'):
                name = f[:-12]
                label = name[:-5]
                
                if label in self.labels:
                    self.label_dict[label].add(name)

        return self.label_dict
    
    def get_number_info(self):
        if not self.label_dict:
            self.get_label_dict()
        
        info = ''
        for label in self.label_dict:
            size = len(self.label_dict[label])
            info += '\nLabel {}: {}'.format(label, size)

        return info
           
    def read_tvhi_label_file(self, label_file):
        f = open(label_file, 'r')
        
        action_start = False
        action_end = False
        action = False
        start_index = 0
        end_index = -1
        
        next(f)
        for line in f:
            if line.startswith('#frame'):
                # Check previous frame
                if action == True and action_start == False:
                    action_start = True
                    start_index = frame_index
                elif action == False and action_start == True:
                    if action_end == False:
                        end_index = frame_index - 1
                        action_end = True

                line = line.split()
                frame_index = int(line[1])
                action = False
            else:
                line = line.split()
                action_label = line[4]
                if action_label != 'no_interaction':
                    action = True
        
        # Check the last frame       
        if action == True and action_start == False:
            action_start = True
            start_index = frame_index
        elif action == False and action_start == True:
            #action_start = False
            if action_end == False:
                end_index = frame_index - 1
                action_end = True
             
        if end_index == -1 and action == True:
            end_index = frame_index
            action_end = True
        
        last_frame = frame_index         
        f.close()
            
        return start_index, end_index, last_frame

    
    def get_frame_info(self):
        self.video_frame_dict = {}
        
        if not self.label_dict :
            _ = self.get_label_dict()

        # The length of the time duration before an action starts
        min_length = float('inf')
        max_length = 0
        avg_length = 0
        shortest_video = ''
        longest_video = ''
        n = 0
        
        for label in self.labels:
            for video in sorted(self.label_dict[label]):
                label_file = os.path.join(self.label_dir, video+'.annotations') 
                action_start, action_end, video_end = self.read_tvhi_label_file(label_file)
                self.video_frame_dict[video] = [action_start, action_end, video_end]
                
                if action_end > -1:
                    length = action_start - 0
                else:
                    length = 0
                if length < min_length:
                    min_length = length
                    shortest_video = video
                if length > max_length:
                    max_length = length
                    longest_video = video

                avg_length += length
                n += 1

        
        info = ''
        info += '\nMin Length: {} frames ({})'.format(min_length, shortest_video)
        info += '\nMax Length: {} frmaes ({})'.format(max_length, longest_video)
        info += '\nAvg Length: {} frames (Total {} videos)'.format(avg_length / n, n)
        
        return info

        
if __name__ == '__main__':
    path = '/tmp3/chiawen/TVHI/'
    name = 'TVHI'
    labels = ['handShake', 'highFive', 'hug', 'kiss']

    TVHI_analyzer = DatasetAnalyzer(path, name, labels)

    number_info = TVHI_analyzer.get_number_info()
    print(number_info)


    frame_info = TVHI_analyzer.get_frame_info()
    print(frame_info)

