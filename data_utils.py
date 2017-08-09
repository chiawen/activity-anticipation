from __future__ import division

import os
import csv
import random

import numpy as np


class DataSet(object):
    def __init__(self, data, labels, seq_lengths):
        assert data.shape[0] == labels.shape[0], (
                'data.shape: {} labels.shape: {}'.format(data.shape, labels.shape))
        
        self._data = data
        self._labels = labels
        self._seq_lengths = seq_lengths
        self._num_examples = data.shape[0]
        self._index_in_epoch = 0


    def data(self):
        return self._data

    def labels(self):
        return self._labels

    def seq_lengths(self):
        return self._seq_lengths

    def num_examples(self):
        return self._num_examples
    
    def reset_index_in_epoch(self):
        self._index_in_epoch = 0
    
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if start == 0 and shuffle:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            self._seq_lengths = self._seq_lengths[perm]


        if start + batch_size < self._num_examples:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end], self._seq_lengths[start:end]

        else:
            self._index_in_epoch = 0
            return self._data[start:], self._labels[start:], self._seq_lengths[start:]



def dense_to_one_hot(labels_dense, num_classes):
    # Convert class labels from scalars to one-hot vector
   num_labels = labels_dense.shape[0]
   index_offset = np.arange(num_labels) * num_classes
   labels_one_hot = np.zeros((num_labels, num_classes), dtype='float32')
   labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1.0
   
   return labels_one_hot

# Reference: https://github.com/fchollet/keras/blob/master/keras/preprocessing/sequence.py
def pad_sequence(sequence, max_seq_length, truncating='pre', dtype='float32', value=0.0):
    """
    Pad a sequence to the max_seq_length.
    
    If a sequence is shorter than the max_seq_length, 
    add paddings to the end of the sequence.
    If a sequence is longer than the max_seq_length,
    truncate the the beginning or the end of the sequence. 
    """
    
    if not hasattr(sequence, '__len__'):
        raise ValueError('sequence must be iterable.')

    sample_shape = tuple()
    sample_shape = np.asarray(sequence).shape[1:]
    
    length = len(sequence)

    padded_sequence = (np.ones((max_seq_length,) + sample_shape) * value).astype(dtype)

    # truncation
    if truncating == 'pre':
        trunc = sequence[-max_seq_length:]
    elif truncating == 'post':
        trunc = sequence[:max_seq_length]
    else:
        raise ValueError('Truncating type "{}" not understood'.format(truncating))

    # check 'trunc' has expected shape
    trunc = np.asarray(trunc, dtype=dtype)
    if trunc.shape[1:] != sample_shape:
        raise ValueError('Shape of sample {} of sequence is different from expected shape {}'.
                format(trunc.shape[1:], sample_shape))
    
    # padding
    padded_sequence[:len(trunc)] = trunc

    if length > max_seq_length:
        length = max_seq_length

    return padded_sequence, length


def split_train_validation(class_dict, val_ratio):
    train_data = []
    train_labels = []
    val_data = []
    val_labels = []
    split_msg = ''
    
    for label in class_dict:
        data = class_dict[label]
        #val_size = int(len(data) * val_ratio)
        # randomly shuffle data
        #random.shuffle(data) 
        val_size = 20
        val_data = val_data + data[:val_size]
        val_labels = val_labels + [label] * len(data[:val_size])
        train_data = train_data + data[val_size:]
        train_labels = train_labels + [label] * len(data[val_size:])
        
        split_msg += '\nClass {}: training data: {}, validation data: {}'.format(
                    label, len(data[val_size:]), val_size)

    return train_data, train_labels, val_data, val_labels, split_msg

def slice_sequence(sequence, max_seq_length, interval=5):
    length = sequence.shape[0]
    limit = max(length-max_seq_length, 1)
    start_idx = range(0, limit, 5)

    slices = []
    for i in start_idx:
        if i + max_seq_length < length:
            slices.append(sequence[i:i+max_seq_length, :])
        else:
            slices.append(sequence[i:, :])

    return slices

def load_data(data_paths, max_seq_length, labels, trunc=False):

    data = []
    seq_lengths = []
    new_labels = []   
    
    for i, path in enumerate(data_paths):
        sequence = np.load(path)
        label = labels[i]
        
        if trunc == False:
            slices = slice_sequence(sequence, max_seq_length=max_seq_length)
            for s in slices:
                padded_s, s_length = pad_sequence(s, max_seq_length=max_seq_length)
                data.append(padded_s)
                seq_lengths.append(s_length)
                new_labels.append(label) 
        else:
            padded_sequence, seq_length = pad_sequence(sequence, max_seq_length=max_seq_length, truncating='pre')
            data.append(padded_sequence)
            seq_lengths.append(seq_length)
            new_labels.append(label)

    return data, seq_lengths, new_labels 

def sample_sub_sequences(length, num_samples, min_len, max_len):
    max_len = min(length, max_len)
    min_len = min(min_len, max_len)
    sequence = []
    for i in range(num_samples):
        l = random.randint(min_len, max_len)
        start_idx = random.randint(0, length - l)
        end_idx = start_idx + l
        if not (start_idx, end_idx) in sequence:
            sequence.append((start_idx, end_idx))

    return sequence


def multiply_data(sequences, lengths, labels, sample_ratio, extra_samples, min_len=5, max_len=30):
    n = 0
    new_sequences = []
    new_lengths = []
    new_labels = []

    for i, seq in enumerate(sequences):
        n += 1
        label = labels[i]
        length = lengths[i]

        # Add original data to new training samples
        new_sequences.append(seq)
        new_lengths.append(length)
        new_labels.append(label)
        
        # Augment new training samples
        samples = sample_sub_sequences(length, int(sample_ratio[label]*extra_samples),
                min_len, max_len)
        
        for s in samples:
            n += 1
            slice_range = range(s[0], s[1])
            sub_length = s[1] - s[0]
            sub_sequence = seq[slice_range,:]
            padded_sub_sequence, _ = pad_sequence(sub_sequence, max_seq_length=max_len)
            
            new_sequences.append(padded_sub_sequence)
            new_lengths.append(sub_length)
            new_labels.append(label)
    
    return new_sequences, new_lengths, new_labels

def get_class_wise_count(labels):
    counts = {}
    for label in labels:    
        if counts.get(label, 'empty') == 'empty':
            counts[label] = 1
        else:
            counts[label] += 1

    return counts

def get_data_sets(data_file, 
                data_path, 
                one_hot = True, 
                val_ratio = 0.2, 
                max_seq_length = 30,
                data_augmentation = True,
                extra_samples = 4):

    if not 0<= val_ratio < 1.0:
        raise ValueError(
                'Validation ratio should be between 0 and 1.0. Received: {}.'
                .format(val_ratio))
    
    class_dict = {}

    with open(data_file, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            video_id = row['id']
            video_path = os.path.join(data_path, video_id + '-feat.npy')
            label = int(row['label'])
            if class_dict.get(label, 'empty') == 'empty':
                class_dict[label] = [video_path]
            else:
                class_dict[label].append(video_path)
  
    
    train_data_paths, train_labels, val_data_paths, val_labels, split_msg = split_train_validation(class_dict, val_ratio)

    train_data, train_seq_lengths, train_labels = load_data(train_data_paths, max_seq_length, train_labels)
    if data_augmentation:
        class_wise_count = get_class_wise_count(train_labels)
        base_class = max(class_wise_count, key=class_wise_count.get) 
        sample_ratio = {}
        for label in class_dict:
            sample_ratio[label] = class_wise_count[base_class] / class_wise_count[label]
            #print('sample ratio of {} : {}/{}={}'.format(
            #    label, class_wise_count[base_class], class_wise_count[label], sample_ratio[label]))

        train_data, train_seq_lengths, train_labels = multiply_data(train_data, train_seq_lengths, train_labels, sample_ratio, extra_samples)
    
    val_data, val_seq_lengths, val_labels = load_data(val_data_paths, max_seq_length, val_labels, trunc=True)
  
    msg = ''
    counts = get_class_wise_count(train_labels)
    for label in counts:
        msg += '\nClass {}: augmented training data: {}'.format(
                    label, counts[label])
    
    train_data = np.array(train_data, dtype='float32')
    #print('train_data.shape: {}'.format(train_data.shape))
    val_data = np.array(val_data, dtype='float32')
    #print('val_data.shape: {}'.format(val_data.shape))
    train_labels = np.array(train_labels)
    #print('train_labels.shape: {}'.format(train_labels.shape))
    val_labels = np.array(val_labels)
    #print('val_labels.shape: {}'.format(val_labels.shape))
    train_seq_lengths = np.array(train_seq_lengths, dtype='int32')
    val_seq_lengths = np.array(val_seq_lengths, dtype='int32')
    #print('train_seq_lengths.sahpe: {}'.format(train_seq_lengths.shape))
    #print('val_seq_lengths.shape: {}'.format(val_seq_lengths.shape))

    if one_hot:
        train_labels = dense_to_one_hot(train_labels, 4)
        val_labels = dense_to_one_hot(val_labels, 4)
    
    train = DataSet(train_data, train_labels, train_seq_lengths)
    validation = DataSet(val_data, val_labels, val_seq_lengths)
    
    msg = split_msg + msg
    
    return train, validation, msg
    

if __name__ == '__main__':
    train_file = '/tmp3/chiawen/TVHI/train_file.csv'
    train_data_feat = '/tmp3/chiawen/TVHI/features/'
    train, validation, msg = get_data_sets(train_file, train_data_feat, one_hot=True, val_ratio=0.2, max_seq_length=30)
    print('training data size: {}'.format(train.num_examples()))
    print('validation data size: {}'.format(validation.num_examples()))

    print(msg)
 
