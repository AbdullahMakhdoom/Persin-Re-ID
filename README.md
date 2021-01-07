# Person Re-Identification

> FYP BEE-8D, SEECS, NUST, 2K16.

A proof-of-concept of an intelligent surviellance software capable of finding a specific person through a series of surviellance footage.
Unlike face and gait recongition (which rely on facial feaures and walking-style to identify a person). person Re-ID makes uses  of the general appearance of the person.

![](header.png)
## Installation

Download the necessary packages
```sh
# install necessary dependencies
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch torchvision cudatoolkit=9.0 -c pytorc
```

## Usage 

For registering predestrain in surviellance footage, provide path $video_path.
```sh
python3 reid_register.py -v $video_path
```
Gallery folder will contain cropped images pedestrain detected.
"gallery.csv" will consist of rows of (1x512) feature vector from OSnet + UTCtimestamp

For querying predestrain in surviellance footage, provide path $video_path.
```sh
python3 reid_query.py -v $video_path
```
press 's' to track. Select the person to query, then press Enter or Space.
Cropped Images will be saved in Query Person, while "query_features.csv" in the same format as gallery.

## Citation

@article{torchreid,
  title   = {{Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch}},
  author  = {Zhou, Kaiyang and Xiang, Tao},
  journal = {arXiv preprint arXiv:1910.10093},
  year    = {2019}  
  }

@inproceedings{zhou2019osnet,
  title   = {Omni-Scale Feature Learning for Person Re-Identification},
  author  = {Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year    = {2019}
}

@article{zhou2019learning,
  title   = {Learning Generalisable Omni-Scale Representations for Person Re-Identification},
  author  = {Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  journal = {arXiv preprint arXiv:1910.06827},
  year    = {2019}
}
