import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss,use_L1_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_L1_loss)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, g_l1):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake, g_l1),flags) if f]
        return loss_filter

    def initialize(self, opt):

        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        

        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids,opt=opt)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 12
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_L1_loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake', 'G_L1')

            # initialize optimizers
            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))



    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, img_agnostic,pose, warped_c, parse, parse_div,misalign_mask,agnostic_mask,ground_truth_image, infer=False):
        #@ TODO: which loss do we have to use?
        #@ TODO: why do we have to normalize pose? \
        # in SPADE normalizing uniform values like openpose_render lose information

        fake_image =self.netG.forward(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(torch.cat((img_agnostic, pose, warped_c), dim=1), fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(torch.cat((img_agnostic, pose, warped_c), dim=1), ground_truth_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((torch.cat((img_agnostic, pose, warped_c), dim=1), fake_image), dim=1))

        loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.1               
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        Feat_map_list = []
        if not self.opt.no_ganFeat_loss:
            feat_weights = 1.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                D_feature_list = []
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    diff = torch.abs(pred_fake[i][j].cpu().detach() - pred_real[i][j].cpu().detach())
                    diff_up = torch.nn.functional.interpolate(diff, scale_factor=2.0 ** j, mode='bilinear')
                    loss_map = diff_up.sum(1)
                    loss_map = loss_map.detach().cpu().numpy()
                    D_feature_list.append(np.squeeze(loss_map))

                Feat_map_list.append(D_feature_list)
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG,vgg_loss_map_list = self.criterionVGG(fake_image, ground_truth_image)
            loss_G_VGG *= self.opt.lambda_feat

        # heejune added L1 for testing
        loss_G_L1 = 0
        if not self.opt.no_L1_loss:
            L1_loss = torch.nn.L1Loss()
            M_align = parse_div[:,2:3,:,:] - misalign_mask#- parse_div[:,6,:,:]
            M_align = torch.tile(M_align,(1,3,1,1))
            L1_fake_image = fake_image.clone()
            L1_real_image = img_agnostic.clone()
            L1_warp_cloth = warped_c.clone()
            L1_real_image[M_align>0.0] = L1_warp_cloth[M_align>0.0]
            M_generate = agnostic_mask - M_align
            L1_real_image[M_generate>0.0] = L1_fake_image[M_generate>0.0]
            loss_G_L1 = L1_loss(L1_fake_image , L1_real_image) * 4 * self.opt.lambda_feat
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_L1 ), fake_image ] ,vgg_loss_map_list,Feat_map_list

    def inference(self,img_agnostic,pose, warped_c, parse, parse_div,misalign_mask,agnostic_mask):

        with torch.no_grad():
            fake_image = self.netG.forward(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, img_agnostic,pose, warped_c, parse, parse_div,misalign_mask,agnostic_mask):

        return self.inference(img_agnostic,pose, warped_c, parse, parse_div,misalign_mask,agnostic_mask)

        
