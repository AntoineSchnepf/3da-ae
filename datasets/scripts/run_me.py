# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import gdown
import shutil
import tempfile
import subprocess
import argparse

if __name__ == '__main__':

    assert 'DATA_DIR' in os.environ, "Please set the DATA_DIR environment variable to the path of the data directory"
    DATA_DIR = os.environ['DATA_DIR']

    with tempfile.TemporaryDirectory() as working_dir:
        download_name = 'cars_train.zip'
        url = 'https://drive.google.com/uc?id=1bThUNtIHx4xEQyffVBSf82ABDDh2HlFn'
        output_dataset_name = 'cars_128.zip'

        dir_path = os.path.dirname(os.path.realpath(__file__))
        extracted_data_path = os.path.join(working_dir, os.path.splitext(download_name)[0])
        print("Downloading data...")
        zipped_dataset = os.path.join(working_dir, download_name)
        gdown.download(url, zipped_dataset, quiet=False)

        print("Unzipping downloaded data...")
        shutil.unpack_archive(zipped_dataset, DATA_DIR)

        print("Converting camera parameters...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_shapenet_cameras.py')} --source={os.path.join(DATA_DIR, os.path.splitext(download_name)[0])}"
        subprocess.run([cmd], shell=True)

        print("Generating dataset with only views from above...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_shapenet_above.py')} --source={os.path.join(DATA_DIR, os.path.splitext(download_name)[0])}"
        subprocess.run([cmd], shell=True)

        print("Converting camera parameters...")
        cmd = f"python {os.path.join(dir_path, 'preprocess_shapenet_cameras.py')} --source={os.path.join(DATA_DIR, 'cars_above_train')}"
        subprocess.run([cmd], shell=True)


        #print("Creating dataset zip...")
        #cmd = f"python dataset_tool.py"
        #cmd += f" --source {extracted_data_path} --dest {os.path.join('../..', output_dataset_name)} --resolution 128x128"
        #subprocess.run([cmd], shell=True)