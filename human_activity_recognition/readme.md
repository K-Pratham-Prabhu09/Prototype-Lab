your_project_name/
├── data/                   # Your datasets go here
├── src/                    # All your actual Python code!
│   ├── __init__.py         # Makes 'src' an importable package (can be empty)
│   ├── dataset.py          # Code to load and process data
│   ├── model.py            # Your PyTorch neural network classes
│   ├── engine.py           # The training and evaluation loops
│   └── utils.py            # Helper functions (saving/loading, metrics)
├── config.py               # All your hyperparameters in one place
├── .gitignore              # Tells Git to ignore your /data and .venv
├── requirements.txt        # Your dependencies
└── main.py                 # The single file you actually run