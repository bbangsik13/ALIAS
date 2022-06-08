import json
from os import path as osp

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from torch.utils import data
from torchvision import transforms
import cv2
'''


  change 
    1. return Ia (for further processing and GT) 
    2. re-scale pose json if load_width != width of image
    3. use noise label for neck (torso skin)
    4. better agnoistic method for upper body and lower body 

'''

'''

   The input Parse label defintion  CIHP_PGN_LABEL 
                [  `'background',  # -
                    'hat',         # hair
                    'hair',        # -
                    'glove',       # noise
                    'sunglasses',  # face
                    'upperclothes',# -
                    'dress',       # upper?
                    'coat',        # upper?
                    'socks',       # -
                    'pants',       # bottom
                    'tosor-skin',  # background?
                    'scarf',       # noise
                    'skirt',       # bottom
                    'face',        # -
                    'leftArm',     # -
                    'rightArm',    # -
                    'leftLeg',     # -
                    'rightLeg',    # -
                    'leftShoe',    # -
                    'rightShoe']   # -

'''


class inpaint_dataset(data.Dataset):


    def __init__(self, opt):
        super(inpaint_dataset, self).__init__()
        self.opt = opt
        self.useSgt = True
        self.mode = opt.dataset_mode  # train or test
        self.dataset_etri = opt.dataset_etri

        # 1. save options
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.input_height = opt.input_height  # default
        self.input_width = opt.input_width  # default
        self.semantic_nc = opt.semantic_nc
        self.bottom_agnostic = opt.bottom_agnostic  # which part to remove
        self.data_path = osp.join(opt.dataroot, opt.dataset_mode)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 2. load data list
        img_names = []  # models
        c_names = []  # cloths
        with open(osp.join(opt.dataroot,opt.dataset_mode, opt.dataset_list), 'r') as f:
            for line in f.readlines():
                img_name, c_name = line.strip().split()
                img_names.append(img_name)
                if self.dataset_etri:  # etri datset use F & B
                    c_name = c_name.replace('.jpg', '_F.jpg')
                c_names.append(c_name)

        self.img_names = img_names
        self.c_names =c_names
        self.labels = {  ##  compressing CHIP PGN to VITON's ##########
            0: ['background', [0]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14,10]],#temparay put neck in left arm
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

    def get_part_arrays(self, parse, pose_data):

        ''' common processing for Ia and Sa
            separate feet from legs for bottom
            separate hands from arms for upper
        '''

        # 1. separate each labels
        parse_array = np.array(parse)
        parse_array[parse_array == 18] = 16  # left shoe 2 left leg
        parse_array[parse_array == 19] = 17  # right shoe 2 right leg
        parse_background = (parse_array == 0).astype(np.float32)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)  #### Check it is OK
        parse_l_arm = (parse_array == 14).astype(np.float32)
        parse_r_arm = (parse_array == 15).astype(np.float32)
        parse_l_leg = (parse_array == 16).astype(np.float32)
        parse_r_leg = (parse_array == 17).astype(np.float32)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))


        # get parse_hands and arms
        # if not bottom_agnostic
        parse_hands = []
        parse_arms = []
        for parse_id, pose_ids in [(14, [6, 7]), (15, [3, 4])]:
            pt_elbow = pose_data[pose_ids[0]]
            pt_wrist = pose_data[pose_ids[1]]
            vec_arm = (pt_wrist - pt_elbow)
            len_arm = np.linalg.norm(pt_wrist - pt_elbow)
            if len_arm != 0:
                vec_arm /= len_arm
            vec_cut_arm = vec_arm[::-1] * np.array([1, -1])
            if np.sum(parse_array == parse_id) == 0:
                parse_arms.append(np.zeros_like(parse_array))
                parse_hands.append(np.zeros_like(parse_array))
                continue
            parse_arm = 255 * (parse_array == parse_id).astype(np.uint8)
            pt1 = tuple((pt_wrist - 10000 * vec_cut_arm / 2).astype(np.int32))
            pt2 = tuple((pt_wrist + 10000 * vec_cut_arm / 2).astype(np.int32))
            cv2.line(parse_arm, pt1, pt2, color=0, thickness=1)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(parse_arm, connectivity=4)
            for x, y in zip(np.arange(pt_elbow[0], pt_wrist[0] + 2e-8, vec_arm[0] + 1e-8),
                            np.arange(pt_elbow[1], pt_wrist[1] + 2e-8, vec_arm[1] + 1e-8)):
                label_arm = labels[int(y), int(x)]
                if label_arm != 0:
                    break
            if label_arm == 0:
                label_arm = -1
            parse_arm = (labels == label_arm).astype(np.float32)
            parse_hand = (parse_array == parse_id).astype(np.float32) - parse_arm
            parse_arms.append(parse_arm)
            parse_hands.append(parse_hand)

        parse_l_arm, parse_r_arm = parse_arms  # seperate left & right
        parse_l_hand, parse_r_hand = parse_hands  # separate left & right

        # get parse_foot and legs
        # if self.bottom_agnostic
        parse_foot = []
        parse_legs = []
        for parse_id, pose_ids in [(16, [13, 14]), (17, [10, 11])]:
            pt_knee = pose_data[pose_ids[0]]
            pt_ankle = pose_data[pose_ids[1]]
            vec_leg = (pt_ankle - pt_knee)
            len_leg = np.linalg.norm(pt_ankle - pt_knee)
            if len_leg != 0:
                vec_leg /= len_leg
            vec_cut_leg = vec_leg[::-1] * np.array([1, -1])
            if np.sum(parse_array == parse_id) == 0:
                parse_legs.append(np.zeros_like(parse_array))
                parse_foot.append(np.zeros_like(parse_array))
                continue
            parse_leg = 255 * (parse_array == parse_id).astype(np.uint8)
            pt1 = tuple((pt_ankle - 10000 * vec_cut_leg / 2).astype(np.int32))
            pt2 = tuple((pt_ankle + 10000 * vec_cut_leg / 2).astype(np.int32))
            cv2.line(parse_leg, pt1, pt2, color=0, thickness=1)
            retval, labels, stats, centroids = cv2.connectedComponentsWithStats(
                parse_leg, connectivity=4)
            for x, y in zip(np.arange(pt_knee[0], pt_ankle[0] + 2e-8, vec_leg[0] + 1e-8),
                            np.arange(pt_knee[1], pt_ankle[1] + 2e-8, vec_leg[1] + 1e-8)):
                label_leg = labels[int(y), int(x)]
                if label_leg != 0:
                    break
            if label_leg == 0:
                label_leg = -1
            parse_leg = (labels == label_leg).astype(np.float32)
            parse_feet = (parse_array == parse_id).astype(np.float32) - parse_leg
            parse_legs.append(parse_leg)
            parse_foot.append(parse_feet)

        parse_l_leg, parse_r_leg = parse_legs  # separate left & right leg
        parse_l_feet, parse_r_feet = parse_foot  # separate left & right foot

        part_arrays = {
            'parse_background': parse_background,
            'parse_head': parse_head,
            'parse_neck': parse_neck,
            'parse_l_arm': parse_l_arm,
            'parse_r_arm': parse_r_arm,
            'parse_l_hand': parse_l_hand,
            'parse_r_hand': parse_r_hand,
            'parse_l_leg': parse_l_leg,
            'parse_r_leg': parse_r_leg,
            'parse_l_feet': parse_l_feet,
            'parse_r_feet': parse_r_feet,
            'parse_upper': parse_upper,
            'parse_lower': parse_lower
        }

        return part_arrays

    def get_agnostic_mask(self, part_arrays, bottom_agnostic):
        '''
            merge mask regions

        '''
        self.bottom_agnostic = bottom_agnostic
        if self.bottom_agnostic:  # lower
            parse_lower = part_arrays['parse_lower']
            parse_l_leg = part_arrays['parse_l_leg']
            parse_r_leg = part_arrays['parse_r_leg']
            mask_parts = [parse_l_leg, parse_r_leg, parse_lower]
        else:  # upper
            parse_upper = part_arrays['parse_upper']
            parse_neck = part_arrays['parse_neck']
            parse_l_arm = part_arrays['parse_l_arm']
            parse_r_arm = part_arrays['parse_r_arm']
            mask_parts = [parse_neck, parse_l_arm, parse_r_arm, parse_upper]
        return mask_parts

    def get_img_agnostic(self, img, part_arrays):

        ''' converting I to Ia using P
            img :  I (PI.Image)
            parts_arrays:  dictonary for each label regions
            return Ia (PIL.Image)
        '''

        # 0. make a clone
        agnostic = img.copy()
        # 1. get mask list
        mask_parts = self.get_agnostic_mask(part_arrays, self.bottom_agnostic)
        # 2. inpaininting using backgound color

        parse_background = part_arrays['parse_background']

        mask = parse_background < 1.0e-8  # get FG,
        background_inpainted = np.array(img)
        inpainting_color = 128#np.mean(background_inpainted[mask == 0], axis=0)  # get average color of BG
        kernel = np.ones((3, 3), np.uint8)  # the kernel size
        niter = 3
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=niter)  #
        background_inpainted[mask > 0] = inpainting_color  # get average color of BG
        background_inpainted = Image.fromarray(background_inpainted)
        # 3. masking the clothing area
        agnostic_mask = np.zeros((1024, 768))
        for parse in mask_parts:
            # enlarge the mask (the boundary of segmentation is precise)
            parse = cv2.dilate(parse, kernel, iterations=niter)  # make dilation image,
            agnostic_mask += parse
            agnostic.paste(background_inpainted, None, Image.fromarray(np.uint8(parse * 255), 'L'))
        agnostic_mask[agnostic_mask > 0.0] = 1.0

        return agnostic, agnostic_mask

    def __getitem__(self, index):

        ''' indexing operation: called batch_size for each iteration '''

        img_name = self.img_names[index]
        c_name = self.c_names[index]

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose-img', pose_name))  # not json but image
        img_width = pose_rgb.size[0]  # image width in saved file
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose-json', pose_name), 'r') as f:
            pose_label = json.load(f)
            # self.resize_pose(pose_label) # (192x256) to load_width, load_height
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
            if self.load_width != img_width:  # compare only width assuming same aspect ratio
                pose_data[:, :2] = pose_data[:, :2] * (self.load_width / img_width)

        # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'img-parse', parse_name))
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)

        parse_long = torch.from_numpy(np.array(parse)[None]).long()  # None? why long?
        parse_gt_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
        parse_gt_map.scatter_(0, parse_long, 1.0)  # one hot encoding
        new_parse_gt_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(self.labels)):
            for label in self.labels[i][1]:
                new_parse_gt_map[i] += parse_gt_map[label]
        # 유지할 부분, 생성할 부분(옷, 피부)
        # 얼굴은 모자이크 처리 되어 있으므로 피부를 추정하는데 도움이 안 될 것
        if self.bottom_agnostic:
            labels = {
                0: ['background', [0, 1, 2, 3, 9, 10, 11, 12]],
                1: ['skin', [5, 6, 7, 8]],  # ????
                2: ['target', [4]],  # torso?
            }
        else:
            labels = {
                0: ['background', [0, 1, 2, 4, 9, 10, 11, 12]],
                1: ['skin', [5, 6, 7, 8]],  # ????
                2: ['target', [3]],  # torso?
            }
            '''
            0:background
            1:hair
            2:face
            3:torso
            4:pants
            5:left_arm
            6:right_arm
            7:left_leg
            8:right_leg
            9:left_shoe
            10:right_shoe
            11:socks
            12:noise
            '''

        parse_final = torch.zeros(3, self.load_height, self.load_width, dtype=torch.float)

        for j in range(len(labels)):
            for label in labels[j][1]:
                parse_final[j] += new_parse_gt_map[label]

        # load person image
        img = Image.open(osp.join(self.data_path, 'img', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        part_arrays = self.get_part_arrays(parse, pose_data)
        img_agnostic, agnostic_mask = self.get_img_agnostic(img, part_arrays)
        img = self.transform(img)

        img_agnostic = self.transform(img_agnostic)  # [-1,1]
        agnostic_mask = torch.tensor(agnostic_mask[np.newaxis, :, :]).type(torch.float32)

        if self.opt.use_warped_cloth:
            warped_name = img_name.split('.')[0]+"_"+ c_name.split('.')[0]+".npy"
            warped_cloth = torch.tensor(np.load(osp.join(self.opt.warped_cloth_dir,'warped_c',warped_name))[0,:,:,:]).type(torch.float32)
            misalign_mask = torch.tensor(np.load(osp.join(self.opt.warped_cloth_dir,'misalign_mask',warped_name))[0,:,:,:]).type(torch.float32)

        else:
            #conture 구하고 그 점들을 가지고 랜덤 크기의 점 그리기 그리고 그 점들을 빼기
            cloth_mask = ((parse_final[2].numpy() > 0) *255).astype(np.uint8)
            align_mask = cloth_mask.copy()
            for _ in range(10):
                contour, _ = cv2.findContours(align_mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                for i in contour:
                    for j in i:
                        if np.random.rand()<0.5:
                            align_mask[j[0][1]][j[0][0]] = 0

            misalign_mask = (((parse_final[2].numpy()>0)*1 - align_mask)>0)*1
            misalign_mask = torch.tensor(misalign_mask[np.newaxis,:]).type(torch.float32)
            warped_cloth = img * (align_mask > 0)


        #ground truth cloth mask affine(perspective?) transform->compare with original cloth mask -> mask misalign region in random
        '''cloth_mask = ((parse_final[2].numpy() > 0) * 255).astype(np.uint8)
        warped_cloth = np.transpose(((img.numpy()+1)/2*255).astype(np.uint8),(1,2,0))
        scale = 0.025
        s_x, s_y, d_x, d_y, theta = (np.random.rand(5) - 0.5) * 2  # s_x,s_y,d_x,d_y,theta
        s_x = 1 + abs(scale) * s_x * -2
        s_y = 1 + abs(scale) * s_y * -2
        d_x = 0#(cloth_mask.shape[0] * scale * d_x).astype(np.int16)
        d_y = 0#(cloth_mask.shape[1] * scale * d_y).astype(np.int16)
        theta = np.pi * scale * theta
        M = np.array([[1.0, 0.0, -cloth_mask.shape[1] / 2], [0.0, 1.0, -cloth_mask.shape[0] / 2], [0.0, 0.0, 1.0]])
        M = np.array([[s_x, 0.0, 0.0], [0.0, s_y, 0.0], [0.0, 0.0, 1.0]]) @ M
        M = np.array([[np.cos(theta), -np.sin(theta), d_x], [np.sin(theta), np.cos(theta), d_y], [0.0, 0.0, 1.0]]) @ M
        M = np.array([[1.0, 0.0, cloth_mask.shape[1] / 2], [0.0, 1.0, cloth_mask.shape[0] / 2], [0.0, 0.0, 1.0]]) @ M
        M = M[:2, :]
        cloth_mask_affine = cv2.warpAffine(cloth_mask, M, (cloth_mask.shape[1], cloth_mask.shape[0]))
        warped_cloth = cv2.warpAffine(warped_cloth,M,(cloth_mask.shape[1], cloth_mask.shape[0]))
        warped_cloth[cloth_mask_affine == 0] = 128
        misalign_mask = cloth_mask - cloth_mask_affine
        misalign_mask = torch.tensor(misalign_mask[np.newaxis, :, :]/255*2-1).type(torch.float32)
        misalign_mask[misalign_mask<1.0]=0
        warped_cloth = torch.tensor(np.transpose(warped_cloth,(2,0,1))/255*2-1).type(torch.float32)'''
        parse_inpaint = torch.cat((parse_final, agnostic_mask), dim=0)

        result = {
            'img_name': img_name,
            'ground_truth_image': img,
            'img_agnostic': img_agnostic,
            'pose': pose_rgb,
            'warped_c': warped_cloth,
            'parse_div':parse_inpaint,
            'misalign_mask':misalign_mask,
            'index':index
        }


        return result

    def __len__(self):

        return len(self.img_names)


class inpaint_dataloader:
    def __init__(self, opt, dataset):

        super(inpaint_dataloader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
            dataset, batch_size=opt.batchSize, shuffle=(train_sampler is None),
            num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def load_data(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

    def name(self):
        print("inpaint dataset")
        return 0

    def __len__(self):
        return len(self.dataset)


