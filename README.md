# activity-anticipation
This is a Python/Tensorflow implementation of an LSTM-based model for human activity anticipation.

## Requirements
- Python
- NumPy
- Tensorflow 1.0
- scikit-image
- Matplotlib

## Data
We use the [TV Human Interactions (TVHI) dataset](http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html). The dataset consists of people performing four different actions: hand shake, high five, hug, and kissing, with a total of 200 videos (excluding the clips that don't contain any of the interactions).
-  [tv\_human\_interactions\_videos.tar.gz](http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz)
-  [tv\_human\_interactions\_annotations.tar.gz](http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_annotations.tar.gz)
Please extract the above files and store the videos inside the `./videos` directory, annotations inside the `./annotations` directory.<br/>


