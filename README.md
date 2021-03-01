# lensgen_pytorch

This repo contains a Pytorch implementation of my Bachelor's thesis. Originally was made in Keras, but i made a Pytorch version in order to learn this framework.

The original Keras version, thesis, and notebooks will be available soon.

Thesis defense is available on [Youtube](https://www.youtube.com/watch?v=n5wXqbQrQdk&ab_channel=DepartamentodeInform%C3%A1ticaUTFSM)

# Dataset

You can download the Gravitational Lens Finding Challenge dataset from [this link](http://metcalf1.difa.unibo.it/blf-portal_bk/data/SpaceBasedTraining.tar.gz) 

# Instalation
~~~
$ git clone https://github.com/Umauro/lensgen_pytorch.git
$ python venv PATH_TO_ENV
$ PATH_TO_ENV\Scripts\activate
$ pip install -r requirements.txt
~~~

# Training

To train the GAN you need use the `train.py` script. This script will save the trained generator state_dict in the models folder, where its name will be `timestamp_EPOCHS_epochs.pt`. Also, the training losses will be saved in `results/train` folder in csv format.

If you use the `--save_progress` parameter, a GIF image will be saved in the previous folder, composed of fixed generated lens images every 5 epochs.

~~~
usage: train.py [-h] --csv_path CSV_PATH --image_folder_path IMAGE_FOLDER_PATH --epochs EPOCHS --batch_size BATCH_SIZE
                [--save_progress | --no-save_progress]

# Example
$ python train.py --csv_path data/SpaceBasedTraining/classifications.csv 
                  --image_folder_path data/SpaceBasedTraining/Public/Band1
                  --epochs 100
                  --batch_size 64
                  --save_progress
~~~

# Training Results

In order to see training results a Streamlit app is provided. In this app you can select a trained model to see its training losses. Also you can see a SLERP interpolation of the selected generator.

~~~
usage: streamlit run results_app.py
~~~

# Generate Images

To generate new images you need to use the `generate_images.py` script. This scripts will create 2 folders in `OUTPUT_PATH` (`npz` and `png`), and will save the generated images as a numpy array and as a png.

~~~
usage: generate_images.py [-h] --model_path MODEL_PATH --output_path OUTPUT_PATH --n_images N_IMAGES

# Example
$ python generate_images.py --model_path models/10-02-21-231322_100_epochs.pt --output_path output/test --n_images 400

~~~

# TODO
- Check the GAN architecture to match the results of Keras Implementation.
- Add other types of GAN (LSGAN,WGAN, etc).