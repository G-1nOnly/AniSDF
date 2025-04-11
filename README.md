# AniSDF
### [[Paper]](https://arxiv.org/pdf/2410.01202v2) [[Project Page]](https://g-1nonly.github.io/AniSDF_Website/)

> [**AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction**](https://arxiv.org/abs/2410.01202v2),            
> [Jingnan Gao](https://g-1nonly.github.io), Zhuo Chen, [Xiaokang Yang](https://english.seiee.sjtu.edu.cn/english/detail/842_802.htm), [Yichao Yan](https://daodaofr.github.io/)<br>
> **Shanghai Jiao Tong University**<br>
> **ICLR 2025**

Official implementation of "AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction".

<div align="center">
  <img src="assets/Teaser.png"/>
</div><br/>

## Requirements and Environments
### Note
- To utilize multiresolution hash encoding or fully fused networks provided by tiny-cuda-nn, you should have at least an RTX 2080Ti, see [https://github.com/NVlabs/tiny-cuda-nn#requirements](https://github.com/NVlabs/tiny-cuda-nn#requirements) for more details.
- To obtain the best results, an Tesla V100 is highly recommended, though most scenes can be recontructed using an RTX 3090.
- Multi-GPU training is currently not supported on Windows.

### Environments
- Install PyTorch>=1.10 [here](https://pytorch.org/get-started/locally/) based the package management tool you used and your cuda version (older PyTorch versions may work but have not been tested)
- Install tiny-cuda-nn PyTorch extension: `pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`
- `pip install -r requirements.txt`

## Run
### Training on Public Dataset
- Download the NeRF-Synthetic data [here](https://www.matthewtancik.com/nerf) and put it under `load/`. The file structure should be like `load/nerf_synthetic/lego`.
- Download the Shiny-Blender data [here](https://dorverbin.github.io/refnerf/index.html) and put it under `load/`. The file structure should be like `load/refnerf/helmet`.
- Download the Shelly data [here](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/) and put it under `load/`. The file structure should be like `load/shelly_dataset/khady`.
- Download the MipNeRF360 data [here](https://jonbarron.info/mipnerf360/) and put it under `load/`. The file structure should be like `load/360_v2/bicycle`.

Run the launch script with `--train`, specifying the config file, the GPU(s) to be used (GPU 0 will be used by default), and the scene name:
```bash
python launch.py --config configs/anisdf-nerf.yaml --gpu 0 --train dataset.scene=lego tag=example
```
The code snapshots, checkpoints and experiment outputs are saved to `exp/[name]/[tag]@[timestamp]`, and tensorboard logs can be found at `runs/[name]/[tag]@[timestamp]`. You can change any configuration in the YAML file by specifying arguments without `--`, for example:
```bash
python launch.py --config configs/anisdf-nerf.yaml --gpu 0 --train dataset.scene=lego tag=iter50k seed=0 trainer.max_steps=50000
```

### Training on Custom COLMAP Data
To get COLMAP data from custom images, you should have COLMAP installed (see [here](https://colmap.github.io/install.html) for installation instructions). Then put your images in the `images/` folder, and run `scripts/imgs2poses.py` specifying the path containing the `images/` folder. For example:
```bash
python scripts/imgs2poses.py ./load/bmvs_dog # images are in ./load/bmvs_dog/images
```
Existing data following this file structure also works as long as images are store in `images/` and there is a `sparse/` folder for the COLMAP output, for example [the data provided by MipNeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip). An optional `masks/` folder could be provided for object mask supervision. To train on COLMAP data, please refer to the example config files `config/*-colmap.yaml`. Some notes:
- Adapt the `root_dir` and `img_wh` (or `img_downscale`) option in the config file to your data;
- The scene is normalized so that cameras have a minimum distance `1.0` to the center of the scene. Setting `model.radius=1.0` works in most cases. If not, try setting a smaller radius that wraps tightly to your foreground object.
- There are three choices to determine the scene center: `dataset.center_est_method=camera` uses the center of all camera positions as the scene center; `dataset.center_est_method=lookat` assumes the cameras are looking at the same point and calculates an approximate look-at point as the scene center; `dataset.center_est_method=point` uses the center of all points (reconstructed by COLMAP) that are bounded by cameras as the scene center. Please choose an appropriate method according to your capture.

### Experiments Note
- Some thoughts on experiments would be updated here.

### Testing
The training procedure are by default followed by testing, which computes metrics on test data, generates animations and exports the geometry as triangular meshes. If you want to do testing alone, just resume the pretrained model and replace `--train` with `--test`, for example:
```bash
python launch.py --config path/to/your/exp/config/parsed.yaml --resume path/to/your/exp/ckpt/ --gpu 0 --test
```

## BibTeX
```bibtex
@inproceedings{gao2025anisdf,
  title={AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction}, 
  author={Jingnan Gao and Zhuo Chen and Xiaokang Yang and Yichao Yan},
  journal={ICLR},
  year={2025},
}
```

## Acknowledgement
Thanks to Yuanchen Guo for his excellent pipeline [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl). The codebase of AniSDF is built upon this wonderful project.
