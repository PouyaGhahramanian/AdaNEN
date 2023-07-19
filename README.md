# AdaNEN: Adaptive Neural Ensemble Network
This repository contains the data and implementation of the research presented in our paper, "A Novel Neural Ensemble Architecture for On-The-Fly Classification of Evolving Text Streams", submitted to ACM Transactions on Knowledge Discovery from Data (TKDD).
## Abstract
We study on-the-fly classification of evolving text streams in which the relation between the input data target labels changes over time—i.e. “concept drift”. These variations decrease the model’s performance, as predictions become less accurate over-time and they necessitate a more adaptable system. While most studies focus on concept drift detection and handling with ensemble approaches, the application of neural models in this area is relatively less studied. We introduce Adaptive Neural Ensemble Network (AdaNEN ), a novel ensemble-based neural approach, capable of handling concept drift in data streams. With our novel architecture, we address some of the problems neural models face when exploited for online adaptive learning environments. Most current studies address concept drift detection and handling in numerical streams, and the evolving text stream classification remains relatively unexplored. We hypothesize that the lack of public and large-scale experimental data could be one reason. To this end, we propose a method based on an existing approach for generating evolving text streams by introducing various types of concept drifts to real-world text datasets. We provide an extensive evaluation of our proposed approach using 12 state-of-the-art baselines and 13 datasets. We first evaluate concept drift handling capability of AdaNEN and the baseline models on evolving numerical streams; this aims to demonstrate the concept drift handling capabilities of our method on a general spectrum and motivate its use in evolving text streams. The models are then evaluated in evolving text stream classification. Our experimental results show that AdaNEN consistently outperforms the existing approaches in terms of predictive performance with conservative efficiency.
## Code & Data
The code and data is organized as follows:
- `AdaNEN.py`: implementation of our proposed Adaptive Neural Ensemble Network.
- `dataset_modifier`: procedure to preprocess the datasets and create data streams.
- `exp.py`: runs AdaNEN and the baseline models on the numerical and text streams, and stores the accuracy and runtime results.
- `plotter.py`: plots the prequential accuracy results, and outputs the overall accuracy and runtime results.
- `baselines/`: implementation of the GOOWE, HBP, Adam, and SGD baseline models.
- `data/`: contains the evolving numerical and text streams used in our experiments.
- `results/`: directory to store the results of the experiments.
## Prerequisites & Setup
To run AdaNEN on your system, you will need the following Python libraries/frameworks:
- [Numpy](https://numpy.org/)
- [Pytorch](https://pytorch.org/)  
To run the experiments with the AdaNEN and the baseline methods, you need the following packages installed:
- [pandas](https://pypi.org/project/pandas/)
- [scikit-learn](https://scikit-learn.org/)
- [scikit-multiflow](https://scikit-multiflow.github.io/)

You can also set up a Python environment to run the code in this repository by using the provided `requirements.txt` file. This can be achieved by running the following command: pip install -r requirements.txt.
## Usage & Experiments
You can use AdaNEN and initiate an instance of it as follows:
model = AdaNEN(feature_size, arch, num_classes, etha, p_drop, betha, s, num_outs, lrs, optimizer)
Parameters:
- feature_size: number of input feature size. Default: 300
- arch: architecture of the hidden layers, in the following format: [hidden_layer_1_size, hidden_layer_2_size, ..., hidden_layer_n_size]. Default: [64, 32, 16]
- num_classes: number of the output classes. Default: 2
- etha: learning rate of the hidden layers. Default: 1e-3
- p_drop: dropout probability in the concatenation layer. Default: 0.2
- betha: penalization parameter used to update ensemble weights. Default: 0.85
- s: regularization parameter for the ensemble method that prevents assigning near zero weights. Default: 0.2
- num_outs: number of output layers. Default: 3
- lrs: learning rates of the output layers. Default: [1e-3, 1e-2, 1e-1]
- optimizer: optimization method used to update parameters of the hidden layers. Default: rmsprop
    
### Sample AdaNEN Run
```python3
from AdaNEN import AdaNEN
from sklearn.datasets import make_blobs
data_size = 1000
input, labels = make_blobs(n_samples = data_size, centers = 2, n_features = 300, random_state = 0)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
classifier = AdaNEN(feature_size = 300, arch = [64, 32], num_classes = 2, etha = 1e-5,
             betha = 0.8, s = 0.2, num_outs = 3, lrs = [1e-3, 1e-2, 1e-1], optimizer = 'rmsprop')
classifier.to(device)
preds = np.zeros(data_size)
for i in range(data_size):
    print('Data Instance: ', i+1)
    preds[i] = classifier.predict(input[i].reshape(1, -1))
    print('Loss: ', classifier.partial_fit(input, labels))
    accuracy = np.sum(labels[:i] == preds[:i])/(i+1) * 100
    print('Prequential Accuracy: ', accuracy)
    print('Ensemble weights: ', classifier.get_weights())
    print('=================================================')
print('Final ensemble weights: ', classifier.get_weights())
accuracy = np.sum(labels == preds)/data_size * 100
print('Overall Accuracy: ', accuracy)
print('=================================================')
```
### Running the experiments
To reproduce the results, follow these steps:
1. Step 1: For the NYT dataset, run dataset modifier as python3 dataset_modifier.py -d [dataset name] -o [output stream name] to construct the stream from the dataset. For other streams, skip this step.
2. Step 2: run exp.py with the following parameters to reproduce the experiments.
    - -s or --stream: stream name, default: nyt
    - -d or --dataset_type: text or numerical
    - -e or --embedding_type: type of the embedding method to be used, bert or w2v
    - -o or --results_file: sub-directory in the `results` folder to store the results
    - -w or --eval_window: evaluation window size
    - -p or --sample_size: portion of data to run the experiment on. If not specified, the entire data will be used
    - -x or --exclude_models: list of the models to be excluded from the experiment
4. Step 3: run the plotter module to show the overall accuracy and runtime results, and generate the prequential accuracy over time plots. Use the following arguments to run plotter.py.
    - -p or --results_path: path to the results directory containing the accuracy and runtime results
    - -o or --output_file: file name to store the generated prequential accuracy plot
    - -w or --window_size: number of window sizes in the prequential accuracy plot
### Sample Experiments Run
The following command runs an experiment on the first 1000 samples of the NYT stream, excludes the AEE and HAT models, and stores the results in the `results/nyt_results` directory.
```python3
python3 exp.py -s nyt -d text -o nyt_results -w 10 -p 1000 -x AEE HAT
```
### 20NG-abrupt, 20NG-gradual, AGNews-abrupt, and AGNews-gradual Evolving Text Streams

Our constructed 20NG-Ab and 20NG-Gr evolving text streams can be accessed from the data directory of this repository. AGNews streams are accessible from the following links.

| Data Stream | AGNews-Ab (W2V embedding)                                                                               | AGNews-Gr (W2V embedding)                                                                               | AGNews-Ab (BERT embedding)                                                                              | AGNews-Gr (BERT embedding)                                                                              |
|-------------|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------:|
| Access Link | [Google Drive Link](https://drive.google.com/file/d/1ZGEP6BxB-rbMZGhQI2cog2CVeve37pl5/view?usp=sharing) | [Google Drive Link](https://drive.google.com/file/d/1urTmlKGqlK5g5hPMbrnuaP14GKOgycoe/view?usp=sharing) | [Google Drive Link](https://drive.google.com/file/d/1cwMFUdp-zNRuxNfDzjF7hcFPPCMHtv9Q/view?usp=sharing) | [Google Drive Link](https://drive.google.com/file/d/1GjiwgzQfnDuZKy8nrdv9ytUMURg-ZoP9/view?usp=sharing) |


There are around 15.1K data instances in the 20NG, and 127.6K in the AGNews text streams. In both data streams, the first 300 features represent the document embeddings, and the last feature is the label. The 20NG data stream can be used with _pickle_ library in Python as follows:
```python3
import pickle
with open('data/20ng_abrupt.data', 'rb') as f:
    stream = pickle.load(f)
```
The concept drift distribution and the drift intensity values for the 20NG streams are shown in the following table.

|      | 3.5K | 4K | 4.5K | 5K | 5.5K | 6K | 6.5K | 7K  | 7.5K | 8K | 8.5K | 9K  | 9.5K | 10K | 10.5K | 11K | 11.5K | 15.1K |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Comp | +    | +  | -    | -  | -    | -  | -    | +   | +    | +  | -    | -   | -    | +   | +     | +   | +     | +     |
| Rec  | -    | -  | +    | -  | -    | -  | +    | +   | -    | -  | +    | +   | +    | -   | +     | +   | -     | -     |
| Sci  | +    | -  | -    | +  | -    | +  | +    | -   | +    | -  | +    | -   | +    | +   | -     | +   | +     | +     |
| Tlk  | -    | -  | -    | -  | +    | +  | -    | -   | -    | +  | -    | +   | +    | +   | +     | -   | -     | -     |
| 1/DI | -    | 50 | 100  | 25 | 200  | 50 | 25   | 150 | 100  | 75 | 250  | 125 | 75   | 50  | 100   | 25  | 50    | -     |

The AGNews streams can be used with _pandas_ library in Python as follows:
```python3
import pandas as pd
stream = pd.read_csv('/data/agnews_abrupt.csv')
```
The concept drift distribution and the drift intensity values for the AGNews streams are shown in the following table.

|      | 10K | 20K | 30K | 40K | 50K | 60K | 70K | 80K  | 90K | 100K | 110K | 127.6K  |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| World     | +    | -  | +    | -  | -    | +  | -    | +   | -    | +  | -    | +   |
| Sports    | +    | -  | -    | +  | +    | -  | +    | -   | -    | +  | -    | +   |
| Business  | -    | +  | +    | +  | -    | -  | +    | +   | +    | -  | +    | -   |
| Sci/Tech  | -    | +  | -    | -  | +    | +  | -    | -   | +    | -  | +    | -   |
| 1/DI | -  | 5000 | 1000  | 500 | 100  | 500 | 2500   | 250 | 2000  | 1000 | 10000  | 500 |

## Citation
<!---If you use AdaNEN in your research, please cite our paper:--->
`To be updated.`
## Contact
If you have any questions or suggestions, feel free to open an issue or pull request or email us at <PouyaGhahramanian@gmail.com>.
