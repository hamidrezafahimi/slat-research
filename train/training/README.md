
## File Structure

```
project/
│
├── main.py          # Entry point for the training process
├── dataset.py       # Handles loading and processing data
├── model.py         # Defines the neural network
├── loss.py          # Implements the custom loss function
├── train.py         # Contains the training logic
├── utils.py         # Utility functions (e.g., checkpoint saving, plotting)
├── r.csv            # Example input data (r)
└── m.csv            # Example mask data (m)
```


## Execution Instructions

Save the .csv files (r.csv, m.csv) in the same directory as the project.
Run the training script:

```bash
python main.py
```

Checkpoints and plots will be generated automatically.