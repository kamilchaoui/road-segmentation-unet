# Road Segmentation

This project performs road segmentation on satellite images using a U-Net model. The code is organized as follows (relevant parts only):

```
├── training/
├── test_set_images/
├── models/
├── predictions/
├── src/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── ensemble_inference.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   └── train.py
├── mask_to_submission.py
├── run.py
└── pyproject.toml
```

- `training/`: Contains the training data.
- `test_set_images/`: Contains the test data.
- `models/`: Stores the trained model weights.
- `predictions/`: Stores the generated prediction masks.
- `src/`: Contains the source code for the project.
  - `augmentations.py`: Defines the data augmentations used during training.
  - `dataset.py`: Defines the PyTorch Dataset for loading the data.
  - `ensemble_inference.py`: Contains the logic for running inference with an ensemble of models.
  - `losses.py`: Defines the loss functions used for training.
  - `metrics.py`: Defines the evaluation metrics used for validation.
  - `model.py`: Defines the U-Net model architecture.
- `mask_to_submission.py`: Converts the prediction masks to the submission CSV format.
- `run.py`: The script to generate the predictions from cached results.
- `pyproject.toml`: The project's dependencies.

## Reproducing the Results

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/CS-433/project-2-rtk2.git
    cd project-2-rtk2
    ```

2.  **Setup a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
    ```

3. **Generate submission:**

    ```bash
    python3 run.py
    python3 mask_to_submission.py
    ```
---

### Training

Download the training data (from [aicrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files)) and place it in the `training/` directory as shown above.

```bash
uv sync
uv run src/train.py
```

This will train an ensemble of 65 models with different seeds for the train/validation split. The trained models will be saved in the `models/` directory. Checkpoints of our trained models are provided on [gdrive](https://drive.google.com/drive/folders/1oZOQsPDQFZ7__IUqNPqCmoFEhzXd8amQ?usp=drive_link).

### Inference

Download the test data (from [aicrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/dataset_files)) and place it in the `test_set_images/` directory.

To run the full inference pipeline and generate the prediction masks:

```bash
uv run src/ensemble_inference.py
uv run run.py
```

This will:
1.  Load the trained models from the `models/` directory.
2.  Run inference on the test images in `test_set_images/`.
3.  Generate prediction masks and save them in the `predictions/` directory.

To generate the final submission file, run the `mask_to_submission.py` script:

```bash
uv run mask_to_submission.py
```

This will convert the prediction masks in the `predictions/` directory to a `submission.csv` file.
