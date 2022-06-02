import json
from os import path as osp

import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
from torch.utils import data
from torchvision import transforms

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


class VITONDataset(data.Dataset):
    ''' @TODO: optimize for train mode and test mode '''

    def __init__(self, opt):
        super(VITONDataset, self).__init__()

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
        self.c_names = dict()
        self.c_names['unpaired'] = c_names  # for testing
        self.labels = {  ##  compressing CHIP PGN to VITON's ##########
            0: ['background', [0, 10]],
            1: ['hair', [1, 2]],
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }

        if False:  # for torso & neck test
            self.labels[0] = ['background', [0]]
            self.labels[12] = ['noise', [3, 10, 11]]

    def get_part_arrays(self, parse, pose_data):

        ''' common processing for Ia and Sa
            separate feet from legs for bottom
            separate hands from arms for upper
        '''

        # 1. separate each labels  @TODO why float?
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

        # @TODO Do we need both foot and arm ?

        # get parse_hands and arms
        # if not bottom_agnostic
        parse_hands = []
        parse_arms = []
        for parse_id, pose_ids in [(14, [6, 7]), (15, [3, 4])]:
            pt_elbow = pose_data[pose_ids[0]]
            pt_wrist = pose_data[pose_ids[1]]
            vec_arm = (pt_wrist - pt_elbow)
            len_arm = np.linalg.norm(pt_wrist - pt_elbow)
            if len_arm != 0:  # @TODO  not 0 but quite small (what if invisible case?)
                vec_arm /= len_arm
            vec_cut_arm = vec_arm[::-1] * np.array([1, -1])
            if np.sum(parse_array == parse_id) == 0:  # @TODO ???
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
            @TODO get a single array?
        '''
        if bottom_agnostic:  # lower
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

    def get_parse_agnostic(self, parse, part_arrays):
        ''' convert S to Sa
            img :  S (PI.Image)
            parts_arrays:  dictonary for each label regions
            return Sa (PIL.Image)
        '''
        # 0. make a clone
        agnostic = parse.copy()
        # 1. get a mask list
        mask_parts = self.get_agnostic_mask(part_arrays, self.bottom_agnostic)
        # 2. masking
        for part in mask_parts:  # masking the regions
            agnostic.paste(0, None, Image.fromarray(np.uint8(part * 255), 'L'))

        return agnostic

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
        # @TODO: but Alias network may use GRAY color for hints for redrawing...
        parse_background = part_arrays['parse_background']
        """        
        mask = np.uint8(255 * (parse_background == 0))
        radius = (np.sum(parse_background == 0) / np.pi)**0.5
        background_inpainted = cv2.inpaint(
            np.array(img), mask, radius, cv2.INPAINT_TELEA)
        """
        mask = parse_background < 1.0e-8  # get FG, @TODO parse_background is float32 ^^;
        background_inpainted = np.array(img)
        inpainting_color = np.mean(background_inpainted[mask == 0], axis=0)  # get average color of BG
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
        from matplotlib import pyplot as plt
        # plt.imshow(agnostic_mask),plt.title('agnostic_mask'),plt.show()
        """ 
        # for checking 
        import matplotlib.pyplot as plt
        plt.title("img_agnostic")
        plt.subplot(1, 2, 1), plt.imshow(np.array(img)),
        plt.subplot(1, 2, 2), plt.imshow(np.array(agnostic)), plt.show()
        """

        return agnostic, agnostic_mask

    def __getitem__(self, index):

        ''' indexing operation: called batch_size for each iteration '''

        img_name = self.img_names[index]
        c_name = {}
        c = {}
        cm = {}
        for key in self.c_names:  # 'upaired' for test and  ('paried' for training? )
            c_name[key] = self.c_names[key][index]
            c[key] = Image.open(osp.join(self.data_path, 'cloth', c_name[key])).convert('RGB')  # C
            c[key] = transforms.Resize(self.load_width, interpolation=2)(c[key])  # size to load_width, load_height?
            if not self.dataset_etri:
                cm[key] = Image.open(osp.join(self.data_path, 'cloth_mask', c_name[key]))  # Mc
            else:
                cm[key] = Image.open(osp.join(self.data_path, 'cloth_mask', c_name[key].replace('jpg', 'png')))  # Mc
            cm[key] = transforms.Resize(self.load_width, interpolation=0)(cm[key])  # size to load_width, load_height?

            c[key] = self.transform(c[key])  # [-1,1],  tensor and normalize
            cm_array = np.array(cm[key])  # type of cm[key]?
            cm_array = (cm_array >= 128).astype(np.float32)  # is range [0, 255]?
            cm[key] = torch.from_numpy(cm_array)  # [0,1]
            cm[key].unsqueeze_(0)  # in-memory

        # load pose image
        pose_name = img_name.replace('.jpg', '_rendered.png')
        pose_rgb = Image.open(osp.join(self.data_path, 'openpose_img', pose_name))  # not json but image
        img_width = pose_rgb.size[0]  # image width in saved file
        pose_rgb = transforms.Resize(self.load_width, interpolation=2)(pose_rgb)
        pose_rgb = self.transform(pose_rgb)  # [-1,1]
        if False:  ### AHN: to  test the effects of rgb in pose image
            # pose_rgb[:,:,:] = 0.0 # black out pose
            pose_rgb[:, :, :] = (pose_rgb[0, :, :] + pose_rgb[1, :, :] + pose_rgb[2, :, :]) / 3.0  # rgb to gray pose

        pose_name = img_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'openpose_json', pose_name), 'r') as f:
            pose_label = json.load(f)
            # self.resize_pose(pose_label) # (192x256) to load_width, load_height
            pose_data = pose_label['people'][0]['pose_keypoints_2d']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
            if self.load_width != img_width:  # compare only width assuming same aspect ratio
                pose_data[:, :2] = pose_data[:, :2] * (self.load_width / img_width)

                # load parsing image
        parse_name = img_name.replace('.jpg', '.png')
        parse = Image.open(osp.join(self.data_path, 'img_parse', parse_name))
        parse = transforms.Resize(self.load_width, interpolation=0)(parse)
        part_arrays = self.get_part_arrays(parse, pose_data)
        parse_agnostic = self.get_parse_agnostic(parse, part_arrays)
        parse_agnostic = torch.from_numpy(np.array(parse_agnostic)[None]).long()  # None? why long?

        parse_agnostic_map = torch.zeros(20, self.load_height, self.load_width,
                                         dtype=torch.float)  # 20 labels input CHIP PGN
        parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)  # one-hot-encoding
        new_parse_agnostic_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
        for i in range(len(self.labels)):  # merging labels into 20 to 13
            for label in self.labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]

        ######## AHN: making one-hot-ended S (nc = 13 labels)####
        if self.useSgt:
            parse_long = torch.from_numpy(np.array(parse)[None]).long()  # None? why long?
            parse_gt_map = torch.zeros(20, self.load_height, self.load_width, dtype=torch.float)
            parse_gt_map.scatter_(0, parse_long, 1.0)  # one hot encoding
            new_parse_gt_map = torch.zeros(self.semantic_nc, self.load_height, self.load_width, dtype=torch.float)
            for i in range(len(self.labels)):
                for label in self.labels[i][1]:
                    new_parse_gt_map[i] += parse_gt_map[label]

        # load person image
        img = Image.open(osp.join(self.data_path, 'img', img_name))
        img = transforms.Resize(self.load_width, interpolation=2)(img)
        img_agnostic, agnostic_mask = self.get_img_agnostic(img, part_arrays)
        img = self.transform(img)
        # agnostic_mask = self.transform(agnostic_mask)
        img_agnostic = self.transform(img_agnostic)  # [-1,1]
        agnostic_mask = torch.tensor(agnostic_mask[np.newaxis, :, :])
        # print(agnostic_mask.size())
        result = {
            'img_name': img_name,
            'c_name': c_name,
            'img': img,
            'img_agnostic': img_agnostic,
            'parse_agnostic': new_parse_agnostic_map,
            'pose': pose_rgb,
            'cloth': c,
            'cloth_mask': cm,
            'agnostic_mask': agnostic_mask
        }
        if self.useSgt:
            result['parse_gt'] = new_parse_gt_map  #### AHN: for training  ####

        return result

    def __len__(self):
        return len(self.img_names)

    '''
    # now we use simpler ways 
    def resize_pose(self, pose):

        if (self.load_width == self.input_width) and (self.load_height == self.input_height):
            return  # nothing to do 


        peoples = pose['people']
        for people in peoples:
            for keys in ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]:
                keypoints = people[keys]
                assert(len(keypoints)%3 == 0)  # x, y, confidence
                for i in range(len(keypoints)//3):
                    keypoints[3*i] *= (self.load_width/self.input_width)  # horizontal
                    keypoints[3*i+1] *= (self.load_height/self.input_height)  # vertical
                    # not chnage confidence value

                     def get_parse_agnostic_old(self, parse, pose_data):

        # convert S to Sa using P 

        parse_array = np.array(parse)
        parse_upper = ((parse_array == 5).astype(np.float32) +
                       (parse_array == 6).astype(np.float32) +
                       (parse_array == 7).astype(np.float32))
        parse_neck = (parse_array == 10).astype(np.float32)

        r = 10
        agnostic = parse.copy()

        # mask arms
        for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
            mask_arm = Image.new('L', (self.load_width, self.load_height), 'black')
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            i_prev = pose_ids[0]
            for i in pose_ids[1:]:
                if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
                pointx, pointy = pose_data[i]
                radius = r*4 if i == pose_ids[-1] else r*15
                mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
                i_prev = i
            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))

        # mask torso & neck
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))

        return agnostic

    def get_img_agnostic_old(self, img, parse, pose_data):

        # converting I to Ia using P 

        parse_array = np.array(parse)
        parse_head = ((parse_array == 4).astype(np.float32) +
                      (parse_array == 13).astype(np.float32))
        parse_lower = ((parse_array == 9).astype(np.float32) +
                       (parse_array == 12).astype(np.float32) +
                       (parse_array == 16).astype(np.float32) +
                       (parse_array == 17).astype(np.float32) +
                       (parse_array == 18).astype(np.float32) +
                       (parse_array == 19).astype(np.float32))

        r = 10 #20
        agnostic = img.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

        return agnostic

    '''


class VITONDataLoader:
    def __init__(self, opt, dataset):
        super(VITONDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
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
        print("StyleViton dataset")
        return 0

    def __len__(self):
        print(len(self.dataset))
        return 0


