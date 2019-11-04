# CenterNet for Lidar Detection
Object detection, 3D detection, and pose estimation using center point detection:
![](readme/fig2.png)
> [**Objects as Points**](http://arxiv.org/abs/1904.07850),            
> Xingyi Zhou, Dequan Wang, Philipp Kr&auml;henb&uuml;hl,        
> *arXiv technical report ([arXiv 1904.07850](http://arxiv.org/abs/1904.07850))*         


## Abstract 
Modify the centernet code for lidar detection. I have got a better performance than pointpillar-based detection model using our  dataset.

## Main results
| Backbone |  Task         |   Recall       | Precision|
|--------------|-------------|--------------|-------------|
|DLA-34        | Lidar Det| 0.89              | 0.83           | 
## Installation
Requirement:

pytorch

spconv  https://github.com/traveller59/spconv (just use the voxel generate operator)
## Use CenterNet For Lidar Detection

Prepare detection data, run:

~~~
python tools/create_kitti_info_file.py create_kitti_info_file /data/object3d/  /data/object3d
~~~
This command will generate traning data info files.
Modify the opts.py, and set opt.root_path to your datapath.
Train the CenterNet model, run:
~~~
sh experiments/car.sh
~~~
Evaluate the model, run:
~~~
cd src; python eval.py cardet
~~~
The output will look like
<p align="center"> <img src='readme/det3.png' align="center" height="230px"> </p>
## TensorRT deploy
| Backbone     | FP32         | FP16           |INT8|
|-----------------|------------|--------------|------|
|DLA-34             | 10-13ms | 7-10ms     |   ---  |

pytorch to onnx to tensorRT code will coming soon

## Citation

If you find this project useful for your research, please use the following BibTeX entry.

    @inproceedings{zhou2019objects,
      title={Objects as Points},
      author={Zhou, Xingyi and Wang, Dequan and Kr{\"a}henb{\"u}hl, Philipp},
      booktitle={arXiv preprint arXiv:1904.07850},
      year={2019}
    }
