# BHAGAVADGITA-GPT

This repository contains code for training a GPT Language Model on the Bhagavad Gita text. The model is trained on character-level encoded data.

## Project Structure

* **Data**
    * **CharLevelEncoded**: This directory stores the preprocessed data for the model.
        * `meta.pkl`: Stores information about the character vocabulary and mapping functions.
        * `train.bin`: Encoded training data in binary format.
        * `val.bin`: Encoded validation data in binary format.
    * **Bhagwad_Gita.csv**: (Optional) The raw Bhagavad Gita data in CSV format.
    * **input.txt**: (Optional) The raw Bhagavad Gita text file.
* **DataPrep**
    * **BPE**: Scripts for Byte Pair Encoding (if used).
    * **charLevel**: This directory focuses on character-level preprocessing.
        * `prepare.py`: Script for preparing the character-level encoded data.
        * `readme.md`: (You are here!) This file describes the project.
* **gitignore**: A file specifying files to be ignored by Git version control.
* **charLevel.py**: Script for training the GPT language model.
* **data_cleaning.ipynb**: (Optional) Jupyter notebook for data cleaning (if used).

## Training the Model

The training script is located in `charLevel.py`. It performs the following steps:

1. Loads the preprocessed data from `Data/CharLevelEncoded`.
2. Defines the GPT model architecture with hyperparameters.
3. Trains the model on the training data and evaluates it on the validation data.
4. Generates text samples from the trained model.

## Generating Text

The trained model can be used to generate new text that resembles the Bhagavad Gita. You can find an example of generating text in `charLevel.py`.

## Dependencies

This project requires the following Python libraries:

* torch
* tqdm
* numpy
* pickle
* os

## How to Run

1. Install the required libraries (`pip install torch tqdm numpy pickle`).
2. Download the Bhagavad Gita text data (if not already available).
3. Preprocess the data using `prepare.py` (if not already done).
4. Train the model using `charLevel.py`.

## Contributing

We welcome contributions to this project! Please feel free to fork the repository and submit pull requests for any improvements or new features.