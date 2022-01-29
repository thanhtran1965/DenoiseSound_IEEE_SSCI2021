# Sound Separation to Improve Sound Classification

## Software requirements:
- librosa, matplotlib, numpy, pandas, sklearn. To download the dependencies: **pip3 install -r requirements.txt**

- MATLAB 2020b.

## Dataset:
- Audio: The raw audio selected from the FSDnoisy18K dataset for evaluating the proposed method. [Download](https://khoavanhoceduvn-my.sharepoint.com/:u:/g/personal/2606_elibrary_su/ET3WCMLLCahIpQpXgKqVxTEBOkDlv2CB4aBjuojr3_t6Dg?e=YOGE5G)

- One_STFT: This dataset consists of 20 classes where each audio per class is extracted to only 1 Short-Time Fourier Transform (STFT). [Download](https://khoavanhoceduvn-my.sharepoint.com/:u:/g/personal/2606_elibrary_su/EVR4dJMDa3VErHXksQbFjxkBdEFIcmBmDCp-K0RUHPsllw?e=QAXjOm)

- Separate_STFT_addNoise_Class: This dataset consists of 21 classes that are 20 classes selected from the original data and additional noisy class. Each audio per class is separated into multiple STFT frames and manually labelled as clean label (the original label of class) and noisy label (that is merged to construct the noisy class). [Download](https://khoavanhoceduvn-my.sharepoint.com/:u:/g/personal/2606_elibrary_su/Ef7nlkv5GjlPmeT9Yo3DaFYBUwZD78EPFC9EAE3S5U6u2w?e=724cFs)

### Note: All above datasets need to download and extract to the [data](https://github.com/nhattruongpham/soundSepsound/tree/main/data) folder.

## Usage:
- First, clone the repository locally:
```
git clone https://github.com/nhattruongpham/soundSepsound.git
```

- Run ```TransferLearning.mlx``` to reproduce the experiments and results with pre-trained CNNs.

- Run ```ReproduceFSDNoisy18k.m``` to reproduce the experiment and result with the proposed network in [1].

- Run ```SFTF_Extractor.ipynb``` to extract STFT features (if any).


## Citation
If you use this code/data or part of it, please cite the following paper:
```
@inproceedings{tran2021separate,
  title={Separate Sound into STFT Frames to Eliminate Sound Noise Frames in Sound Classification},
  author={Tran, Thanh and Huy, Kien Bui and Pham, Nhat Truong and Carrat{\`u}, Marco and Liguori, Consolatina and Lundgren, Jan},
  booktitle={2021 IEEE Symposium Series on Computational Intelligence (SSCI)},
  pages={1--7},
  year={2021},
  organization={IEEE}
}
```