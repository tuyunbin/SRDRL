# Semantic Relation-aware Difference Representation Learning for Change Captioning
This package contains the accompanying code for the following paper:

Tu, Yunbin, et al. ["Semantic Relation-aware Difference Representation Learning for Change Captioning."](https://aclanthology.org/2021.findings-acl.6.pdf), which has appeared as long paper in the Findings of the ACL, 2021. 

## We illustrate the training details as follows:


## Installation

1. Clone this repository

2. cd SRDRL

3. Make virtual environment with Python 3.5 (e.g., conda create -n change python=3.5)

4. Install requirements (pip install -r requirements.txt)

5. [Clone COCO caption eval tools with Python 3.](https://gitee.com/tuyunbin/coco-caption_python3.git)

## Data

1. Download data from here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view?usp=sharing)
```
python google_drive.py 1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe clevr_change.tar.gz
tar -xzvf clevr_change.tar.gz
```
Extracting this file will create `data` directory and fill it up with CLEVR-Change dataset.

2. Preprocess data

The preprocessed data here: [google drive link](https://drive.google.com/file/d/1FA9mYGIoQ_DvprP6rtdEve921UXewSGF/view?usp=sharing).
You can skip the procedures explained below and just download them using the following command:
```
python google_drive.py 1FA9mYGIoQ_DvprP6rtdEve921UXewSGF ./data/clevr_change_features.tar.gz
cd data
tar -xzvf clevr_change_features.tar.gz
```

* Extract visual features using ImageNet pretrained ResNet-101:
```
# processing default images
python scripts/extract_features.py --input_image_dir ./data/images --output_dir ./data/features --batch_size 128

# processing semantically changes images
python scripts/extract_features.py --input_image_dir ./data/sc_images --output_dir ./data/sc_features --batch_size 128

# processing distractor images
python scripts/extract_features.py --input_image_dir ./data/nsc_images --output_dir ./data/nsc_features --batch_size 128
```

* Build vocab and label files using caption annotations:
```
python scripts/preprocess_captions_pos.py --input_captions_json ./data/change_captions.json --input_neg_captions_json ./data/no_change_captions.json --input_image_dir ./data/images --input_pos ./data/pos_token.pkl --split_json ./data/splits.json --output_vocab_json ./data/vocab.json --output_h5 ./data/labels.h5
```

## Training
To train the proposed method, run the following commands:
```
# create a directory or a symlink to save the experiments logs/snapshots etc.
mkdir experiments
# OR
ln -s $PATH_TO_DIR$ experiments

# this will start the visdom server for logging
# start the server on a tmux session since the server needs to be up during training
python -m visdom.server

# start training
python train.py --cfg configs/dynamic/dynamic_change_pos.yaml 
```

The training script runs the model on the validation dataset every snapshot iteration and one can save the visualizations of dual attentions and dynamic attentions using the flag `visualize`:
```
python train.py --cfg configs/dynamic/dynamic_change_pos.yaml --visualize
```

One can also control the strength of entropy regularization over the dynamic attention weights using the flag `entropy_weight`:
```
python train.py --cfg configs/dynamic/dynamic_change_pos.yaml --visualize --entropy_weight 0.0001
```

## Testing/Inference
To test/run inference on the test dataset, run the following command
```
python test.py --cfg configs/dynamic/dynamic_change_pos.yaml --visualize --snapshot 6000 --gpu 1
```
The command above will take the model snapshot at 6000th iteration and run inference using GPU ID 1, saving visualizations as well.

## Evaluation
* Caption evaluation

To evaluate captions, we need to first reformat the caption annotations into COCO eval tool format (only need to run this once). After setting up the COCO caption eval tools, make sure to modify `utils/eval_utils.py` so that the `COCO_PATH` variable points to the COCO eval tool repository. Then, run the following command:
```
python utils/eval_utils.py
```

After the format is ready, run the following command to run evaluation:
```
# This will run evaluation on the results generated from the validation set and print the best results
python evaluate.py --results_dir ./experiments/dynamic_SRDRL/eval_sents --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```

Once the best model is found on the validation set, you can run inference on test set for that specific model using the command exlpained in the `Testing/Inference` section and then finally evaluate on test set:
```
python evaluate.py --results_dir ./experiments/dynamic_SRDRL/test_output/captions --anno ./data/total_change_captions_reformat.json --type_file ./data/type_mapping.json
```
The results are saved in `./experiments/dynamic_SRDRL/test_output/captions/eval_results.txt`

If you find this helps your research, please consider citing:
```
@inproceedings{tu2021semantic,
  title={Semantic Relation-aware Difference Representation Learning for Change Captioning},
  author={Tu, Yunbin and Yao, Tingting and Li, Liang and Lou, Jiedong and Gao, Shengxiang and Yu, Zhengtao and Yan, Chenggang},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={63--73},
  year={2021}
}
```


## Contact
My email is tuyunbin1995@foxmail.com

Any discussions and suggestions are welcome!


## Acknowledgement
This work and code are inspired by [Robust Change Captioning](https://github.com/Seth-Park/RobustChangeCaptioning). Thanks for their solid work!
