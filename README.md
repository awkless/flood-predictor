# CSCI 4405/5405 Project

Flash flood predictor.

## Install

Firstly, install pipx in order to install UV. Once pipx is installed run the
following:

```
pip install uv
```

Secondly, create a virtual environment through UV and source it:

```
uv venv
source .venv/bin/activate
```

Lastly, build the application like so:

```
uv pip install -e .
```

Now the `flood-predictor-app` GUI is available in the virtual environment to predict flash flood probability and risk level, and be run as:
```
flood-predictor-app /path/to/gage-height.csv /path/to/discharge.csv
```

Also, the `flood-predictor` CLI is available in the virtual environment and can be used to train the model:
```
flood-predictor /path/to/gage-height.csv /path/to/discharge.csv
```

## Usage

### Training the model
python flood\_prediction.py <gage_height_data.csv> <discharge_data.csv>

### Run the GUI
python app.py /path/to/gage-height.csv /path/to/discharge.csv


### Notes

- Just uses the normal csv files and merges gage height and discharge data based
  on timestamps. 
- Uses a threshold of 2.2 feet for a flash flood. This is taken from the peaks
  data and is just the lowest point for now. 
- Most of the predictions are 0.002% because of how little the data fluctuates.
  It may be beneficial for us to introduce artificial flash flood points to
  allow the model to learn and predict better. 

## Software Architecture

### Data

Features used for prediction:
- gage\_height
- discharge
- gage\_diff (difference between consecutive gage measurements)

### Model

TCN:
- 3 input channels (gage\_height, discharge, gage\_diff)
- Hidden size: 16
- Kernel size: 2
- Output: predicted gage height for the next time step
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate: 0.0003)
- Sequence length: 72 (using the past 72 gage height measurements)

### Training

- Number of epochs: 10   
- Batch size: 32   
- Training data: 80% of the dataset   
- Validation data: 20% of the dataset   

### Evaluation

After training, the model predicts:   
- Gage heights   
- Flash flood probabilities (computed using a sigmoid relative to the threshold)

### Interpreting Outputs

Predicted gage heights: the models water level at the next time step.   Flash
flood probability (%): how likely it is that the gage height exceeds the 2.2 ft
threshold.
- 0% → very unlikely to flood
- 50% → near threshold
- 100% → definitely above threshold (flash flood)

## License

This project uses the MIT license.
