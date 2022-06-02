import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize,normalize_transform
from data.image_folder import make_dataset
from PIL import Image
from glob import glob
import numpy as np
import torch
import cv2

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = os.path.join(opt.dataroot,opt.dataset_mode)#datasets/alias
        ## @TODO : Dangerous way, if one mistake will spoil all and cannot check the error

        ## img_argnostic
        self.img_agnostic_dir = os.path.join(self.root,'img_agnostic')

        self.img_agnostic_paths = sorted(glob(os.path.join(self.img_agnostic_dir,"*")))

        ## pose
        self.pose_dir = os.path.join(self.root, 'pose')
        self.pose_paths = sorted(glob(os.path.join(self.pose_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.pose_paths):
            raise("the data length mismatch")

        ## warped_c
        self.warped_c_dir = os.path.join(self.root, 'warped_c')
        self.warped_c_paths = sorted(glob(os.path.join(self.warped_c_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.warped_c_paths):
            raise("the data length mismatch")

        ## agnostic_mask 
        self.agnostic_mask_dir = os.path.join(self.root, 'agnostic_mask')
        self.agnostic_mask_paths = sorted(glob(os.path.join(self.agnostic_mask_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.agnostic_mask_paths):
            raise("the data length mismatch")
       
        ## parse
        self.parse_dir = os.path.join(self.root, 'parse')
        self.parse_paths = sorted(glob(os.path.join(self.parse_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.parse_paths):
            raise("the data length mismatch")

        ## parse_div
        self.parse_div_dir = os.path.join(self.root, 'parse_div')
        self.parse_div_paths = sorted(glob(os.path.join(self.parse_div_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.parse_div_paths):
            raise("the data length mismatch")

        ## misalign_mask
        self.misalign_mask_dir = os.path.join(self.root, 'misalign_mask')
        self.misalign_mask_paths = sorted(glob(os.path.join(self.misalign_mask_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.misalign_mask_paths):
            raise("the data length mismatch")

        ## ground_truth
        self.ground_truth_dir = os.path.join(self.root, 'ground_truth')
        self.ground_truth_paths = sorted(glob(os.path.join(self.ground_truth_dir,"*")))
        if len(self.img_agnostic_paths) != len(self.ground_truth_paths):
            raise("the data length mismatch")


        self.dataset_size = len(self.img_agnostic_paths)

    def __getitem__(self, index):
        ## img_agnostic
        img_agnostic_path = self.img_agnostic_paths[index]
        img_agnostic = np.load(img_agnostic_path)
        img_agnostic = np.squeeze(img_agnostic)
        img_agnostic_tensor = torch.from_numpy(img_agnostic)

        ## pose
        pose_path = self.pose_paths[index]
        pose = np.load(pose_path)
        pose = np.squeeze(pose)
        pose_tensor = torch.from_numpy(pose)

        ## warped_c
        warped_c_path = self.warped_c_paths[index]
        warped_c = np.load(warped_c_path)
        warped_c = np.squeeze(warped_c)
        warped_c_tensor = torch.from_numpy(warped_c)

        ## agnostic_mask
        agnostic_mask_path = self.agnostic_mask_paths[index]
        agnostic_mask = np.load(agnostic_mask_path)
        
        agnostic_mask = np.squeeze(agnostic_mask)
        agnostic_mask = agnostic_mask[np.newaxis,:]
        #print(agnostic_mask.shape)
        # remove the whilte pixels at boundary 
        #print("type of warped_cm", warped_cm.dtype)
        if False:
            warped_cm  = cv2.erode(warped_cm, np.ones((7,7))) 
        agnostic_mask_tensor = torch.from_numpy(agnostic_mask)
        
        ## parse
        parse_path = self.parse_paths[index]
        parse = np.load(parse_path)
        #print(parse.shape)
        parse = np.squeeze(parse)
        parse_tensor = torch.from_numpy(parse)

        ## parse_div
        parse_div_path = self.parse_div_paths[index]
        parse_div = np.load(parse_div_path)
        #print(parse_div.shape)
        parse_div = np.squeeze(parse_div)
        parse_div_tensor = torch.from_numpy(parse_div)

        ## misalign_mask
        misalign_mask_path = self.misalign_mask_paths[index]
        misalign_mask = np.load(misalign_mask_path)
        #print(misalign_mask.shape)

        misalign_mask = np.squeeze(misalign_mask)
        misalign_mask = misalign_mask[np.newaxis,:]
        #print(misalign_mask.shape)
        #exit(0)
        misalign_mask_tensor = torch.from_numpy(misalign_mask)

        ## ground_truth
        ground_truth_path = self.ground_truth_paths[index]
        ground_truth_image = Image.open(ground_truth_path)
        transform_synthesis_image = normalize_transform()
        ground_truth_image_tensor = transform_synthesis_image(ground_truth_image.convert('RGB'))

        input_dict = {'index': index, 
                'img_agnostic':img_agnostic_tensor,
                'pose':pose_tensor,
                'warped_c':warped_c_tensor,
                'parse':parse_tensor,
                'parse_div':parse_div_tensor,
                'misalign_mask':misalign_mask_tensor,
                'ground_truth_image':ground_truth_image_tensor,
                'agnostic_mask': agnostic_mask_tensor,
                'path':ground_truth_path}

        return input_dict

    def __len__(self):
        return len(self.img_agnostic_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'





