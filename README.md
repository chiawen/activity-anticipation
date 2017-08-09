# activity-anticipation
Human activity anticipation from videos using an LSTM-based model.
![problem statement](assets/problem_statement.png)

## Requirements
- Python
- NumPy
- Tensorflow 1.0
- scikit-image
- Matplotlib

## Data
We use the [TV Human Interactions (TVHI) dataset](http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html). The dataset consists of people performing four different actions: hand shake, high five, hug, and kiss, with a total of 200 videos (excluding the clips that don't contain any of the interactions).
-  [tv\_human\_interactions\_videos.tar.gz](http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_videos.tar.gz)
-  [tv\_human\_interactions\_annotations.tar.gz](http://www.robots.ox.ac.uk/~alonso/data/tv_human_interactions_annotations.tar.gz)
<br/>
Please extract the above files and store the videos inside the `./videos` directory, annotations inside the `./annotations` directory.<br/>

## Model
![model](assets/model.png)
