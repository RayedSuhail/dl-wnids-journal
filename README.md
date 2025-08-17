# Lightweight CNN-Based Wi-Fi Intrusion Detection Using 2D Traffic Representations

This repository will track the code and changes made with respect to the project that Rayed Suhail Ahmad is doing under Dr. Quamar Niyaz.

## Setting up Anaconda Environment
To utilize the code written in this repo, please follow the following instructions for installing Anaconda and creating a conda environment for running the code by utilizing the provided `environment.yml` file. To start, you need to first use this [link](https://www.anaconda.com/download) to download the Anaconda installer. Follow the steps for your system provided in the [installation guide](https://docs.anaconda.com/free/anaconda/install/index.html) to set up Anaconda with a graphical interface. Once you have followed all the steps and successfully installed Anaconda in your system, open the Anaconda Prompt console. Using the console, navigate to where you cloned this repository and type the following command to create a duplicate virtual environment to the one that was used to train the model:

`conda env create -f environment.yml`

Once the process completes, you can activate the environment by running the command:

`conda activate amlenv`

While in this environment you can utilize the same packages that were utilized in creating the model by using the `python` command and running a Python console directly in the Anaconda Prompt. However, for using IDE we suggest `Spyder` as it allows for amazing graphical features like Variable Explorer, Python Terminal Manager, and Code Editor. Spyder can be started using the current conda environment by running the following command:

`spyder`

## Running the script
This section focuses on how to run the scripts within this repo to train your own models on your local system. Please follow the instructions in order:

Create a folder named `datasets` in the repo base directory and place the CSV files downloaded from [here](https://icsdweb.aegean.gr/awid/download-dataset) into a folder named `AWID3`. Choose whichever attacks you wish to train the models on by changing the end of the folders to say `_Same`. The folder should look like the following if you wish to follow our paper:

![Multiple folders corresponding to attack types in AWID3 with '_Same' attached at the end](<images/AWID3 Folders.png>)

Before you run the `update_csv_files.py` script, ensure that the current working directory is set to the `scripts` folder and go over lines 17-18 and lines 65-72. You can run the script multiple times but make sure to comment out lines 100-101 to avoid reprocessing of combined CSV file. The `datasets` folder should like as follows:

![Two folders and multiple variations of CSV files for training-testing-validation data](<images/Post-Script Dataset Folder.png>)

Now that the dataset has been set up properly, go to `model_training.py` and change the number of epochs as per your training on line 17. Additionally, if you wish to skip certain model types or techniques, uncomment lines 55-56 and update the conditions as per requirement.

### This should allow you to replicate the training process for our paper and provide with several trained models for you to use.