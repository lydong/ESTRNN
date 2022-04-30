# ESTRNN for Super Resolution

This work is based on:
Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring (ECCV2020 Spotlight)
[Conference version (old BSD dataset)](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510188.pdf) 
by Zhihang Zhong, Ye Gao, Yinqiang Zheng, Bo Zheng, Imari Sato

## Quick Start

### Prerequisites

- Python 3.6 #3.8 is tested to work
- PyTorch 1.6 with GPU # 1.7.1 is tested to work
- opencv-python
- scikit-image
- lmdb
- thop
- tqdm
- tensorboard

### Data Set
- [REDS](https://seungjunnah.github.io/Datasets/reds.html),
I used "Sharp" + "Low resolution"

### Avaible checkpoints
TO-DO

### Testing/Inference
```bash
python main.py --data_root <Your Root Path> --dataset REDS_png --model ESTRNN_SRx4 --n_blocks 15 --n_features 16
--test_only --test_checkpoint <Your Checkpoint Path>
--num_gpus 2 --batch_size 8 --threads 16
```

Please change parameters according to your hardware. 
Note that for testing, please store your test image sequences in the following way:
"*Your Root Path/REDS_png/valid/valid_sharp/seq num/img num.png*"
and 
"*Your Root Path/REDS_png/valid/valid_sharp_bicubic/X4/seq num/img num.png*"
where "*seq num*" is the index of the sequence, "*img num*" is the index of a image inside. 

For example,
"*Your Root Path/REDS_png/valid/valid_sharp/006/066.png*" and 
"*Your Root Path/REDS_png/valid/valid_sharp/006/066.png*".

Note that this version is hardcoded to inference sequences with 100 images. 
If you want to modify the testing pipeline, please go to function _test_torch_SR in "*train/test.py*"  and help yourself.
When GT is not obtainable, simply comment out those lines that load such image and make comparisons (e.g., PSNR, SSIM).

### Training
Similar to testing, store your training sequences in this way:
"*Your Root Path/REDS_png/train/train_sharp/seq num/img num.png*"
and 
"*Your Root Path/REDS_png/train/train_sharp_bicubic/X4/seq num/img num.png*"

To reproduce our mid-size model, use the following command:
```bash
python main.py --data_root <Your Root Path> --dataset REDS_png --model ESTRNN_SRx4 
--end_epoch 240  --n_blocks 15 --n_features 20 --max_grad 26 --lr 2e-4 --learning_cycle 1
--num_gpus 2 --threads 16 --batch_size 8 
```

## Citing

If you use any part of our code, or ESTRNN and BSD are useful for your research, please consider citing original authors:

```bibtex
@inproceedings{zhong2020efficient,
  title={Efficient spatio-temporal recurrent neural network for video deblurring},
  author={Zhong, Zhihang and Gao, Ye and Zheng, Yinqiang and Zheng, Bo},
  booktitle={European Conference on Computer Vision},
  pages={191--207},
  year={2020},
  organization={Springer}
}

@misc{zhong2021efficient,
      title={Efficient Spatio-Temporal Recurrent Neural Network for Video Deblurring}, 
      author={Zhihang Zhong and Ye Gao and Yinqiang Zheng and Bo Zheng and Imari Sato},
      year={2021},
      eprint={2106.16028},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

```
