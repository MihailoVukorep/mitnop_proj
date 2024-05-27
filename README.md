# Book Archiver - detecting hand written digits and letters from camera/photos

## Setting up the python virtual environment
To set up the python virtual environment run the following command:

```
bash setup-env.sh
```

the script will install all the needed libraries
<br>
use the environment with:

```
source p3env/bin/activate
```

## Download the EMNIST dataset
run the following command to download the EMNIST dataset 
```
bash dldataset.sh
```
after the command finishes you should have the gzip.zip file in the datasets folder
<br>
the script will also extract the zip file
- https://www.nist.gov/itl/products-and-services/emnist-dataset
- https://biometrics.nist.gov/cs_links/EMNIST/Readme.txt
- https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip

## Analyse the downloaded set
```
python dataset_analysis.py
```
![char count image](stats/dataset_unqiue_count_all.png "character count")

## Training the convolutional neural network

```
python train_load_all.py
```
