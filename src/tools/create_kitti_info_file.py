
import numpy as np
import kitti_common as kitti
import fire
from pathlib import Path
import pickle

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line[:-1] for line in lines]

def create_kitti_info_file(data_path, save_path=None, relative_path=True):
    train_img_ids = _read_imageset_file((data_path + "train.txt"))
    val_img_ids = _read_imageset_file((data_path + "val.txt"))
    #test_img_ids = _read_imageset_file(str(imageset_folder / "test.txt"))

    print("Generate info. this may take several minutes.")
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    #_calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    filename = save_path / 'infos_train.pkl'
    print(f"Kitti info train file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    kitti_infos_val = kitti.get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    #_calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)
    filename = save_path / 'infos_val.pkl'
    print(f"Kitti info val file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    filename = save_path / 'infos_trainval.pkl'
    print(f"Kitti info trainval file is saved to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)

if __name__ == '__main__':
    fire.Fire()
