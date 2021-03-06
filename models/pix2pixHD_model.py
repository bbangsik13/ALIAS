import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import cv2

class Pix2PixHDModel(BaseModel):

    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
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
            netD_input_nc = 3+3+3+3+7 #warped_c, pose, image_agnositc, fake_image, parse
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
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss)
            
            self.criterionGAN = networks.GANLoss('hinge', tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

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

    def forward(self, img_agnostic,pose, warped_c, parse, parse_div,misalign_mask,ground_truth_image, infer=False):
        #@ TODO: which loss do we have to use?
        #@ TODO: why do we have to normalize pose? \
        # in SPADE normalizing uniform values like openpose_render lose information
        fake_image =self.netG.forward(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)
        # Fake Detection and Loss

        pred_fake_pool = self.discriminate(torch.cat((parse, pose,img_agnostic, warped_c), dim=1), fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False,for_discriminator=True)

        # Real Detection and Loss        
        pred_real = self.discriminate(torch.cat((parse, pose,img_agnostic, warped_c), dim=1), ground_truth_image)
        loss_D_real = self.criterionGAN(pred_real, True,for_discriminator=True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((torch.cat((parse, pose,img_agnostic, warped_c), dim=1), fake_image), dim=1))

        loss_G_GAN = self.criterionGAN(pred_fake, True,for_discriminator=False) #* 0.1
        # GAN feature matching loss
        loss_G_GAN_Feat = 0

        Feat_loss_map = torch.zeros((fake_image.shape[0],self.opt.num_D * (len(pred_fake[0])-1),fake_image.shape[2],fake_image.shape[3])).cuda()

        if not self.opt.no_ganFeat_loss:
            feat_weights = 1.0 #/ (self.opt.n_layers_D + 1)#match with SPADE(unweighted_loss)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                    diff = torch.abs(pred_fake[i][j].detach() - pred_real[i][j].detach()).mean(1) * D_weights * feat_weights * self.opt.lambda_feat
                    diff = torch.unsqueeze(diff,1)
                    diff = torch.nn.functional.interpolate(diff,size=(fake_image.shape[2],fake_image.shape[3]),mode='bilinear')
                    Feat_loss_map[:,(len(pred_fake[i])-1)* i + j,:,:] = diff


        # VGG feature matching loss
        vgg_loss_map = torch.zeros((fake_image.shape[0],5,fake_image.shape[2],fake_image.shape[3])).cuda()
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG,vgg_loss_map = self.criterionVGG(fake_image, ground_truth_image)
            loss_G_VGG *= self.opt.lambda_feat
            vgg_loss_map*= self.opt.lambda_feat


        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake), fake_image ] ,vgg_loss_map,Feat_loss_map

    def inference(self,img_agnostic,pose, warped_c, parse, parse_div,misalign_mask):

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
    def forward(self, img_agnostic,pose, warped_c, parse, parse_div,misalign_mask):

        return self.inference(img_agnostic,pose, warped_c, parse, parse_div,misalign_mask)

        
