import time
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
from subprocess import call
import fractions
def lcm(a,b): return abs(a * b)/fractions.gcd(a,b) if a and b else 0

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer


# 1. option 
opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    try:
        start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    except:
        start_epoch, epoch_iter = 1, 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))        
else:    
    start_epoch, epoch_iter = 1, 0

opt.print_freq = lcm(opt.print_freq, opt.batchSize)    
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# 2. dataset and loader 
data_loader = CreateDataLoader(opt)

if opt.use_previous_network_output:
    dataset = data_loader.load_data()
else:
    dataset = data_loader.data_loader
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# 3. create model 
model = create_model(opt)

# loggers 
visualizer = Visualizer(opt)

# optimizer
optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

#visualize
total_steps = (start_epoch-1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

once = True  # run once
once2 = True

# 4. epoch loop (fixed lr period and 
for epoch in range(start_epoch, opt.niter + opt.niter_stable + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    save_epoch_iter = epoch_iter  # heejun
    for i, data in enumerate(dataset, start=epoch_iter):

        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta
        if opt.inpaint:

            if once:  # checking input
                once = False
                print("Ia:{}".format(data['img_agnostic'].size()), "dtype: ", data['img_agnostic'].dtype,
                      torch.max(data['img_agnostic']), torch.min(data['img_agnostic']))
                print("P:{}".format(data['pose'].size()), "dtype: ", data['pose'].dtype, torch.max(data['pose']),
                      torch.min(data['pose']))
                print("Wc:{}".format(data['warped_c'].size()), "dtype: ", data['warped_c'].dtype,
                      torch.max(data['warped_c']), torch.min(data['warped_c']))
                print("Sdiv:{}".format(data['parse_div'].size()), "dtype: ", data['parse_div'].dtype,
                      torch.max(data['parse_div']), torch.min(data['parse_div']))
                print("Mdiv:{}".format(data['misalign_mask'].size()), "dtype: ", data['misalign_mask'].dtype,
                      torch.max(data['misalign_mask']), torch.min(data['misalign_mask']))
                print("gt:{}".format(data['ground_truth_image'].size()), "dtype: ", data['ground_truth_image'].dtype,
                      torch.max(data['ground_truth_image']), torch.min(data['ground_truth_image']))

            Ia = Variable(data['img_agnostic'])
            P = Variable(data['pose'])
            Wc = Variable(data['warped_c'])
            Sdiv = Variable(data['parse_div'])
            Mdiv = Variable(data['misalign_mask'])
            Igt = Variable(data['ground_truth_image'])

            losses, generated = model(Ia, P, Wc, Sdiv, Mdiv, Igt, infer=save_fake)
        else:
            if once: # checking input
                once = False
                print("Ia:{}".format(data['img_agnostic'].size()),"dtype: ", data['img_agnostic'].dtype,torch.max(data['img_agnostic']),torch.min(data['img_agnostic']))
                print("P:{}".format(data['pose'].size()),"dtype: ", data['pose'].dtype,torch.max(data['pose']),torch.min(data['pose']))
                print("Wc:{}".format(data['warped_c'].size()),"dtype: ", data['warped_c'].dtype,torch.max(data['warped_c']),torch.min(data['warped_c']))
                print("M_a:{}".format(data['agnostic_mask'].size()),"dtype: ", data['agnostic_mask'].dtype,torch.max(data['agnostic_mask']),torch.min(data['agnostic_mask']))
                print("S:{}".format(data['parse'].size()),"dtype: ", data['parse'].dtype,torch.max(data['parse']),torch.min(data['parse']))
                print("Sdiv:{}".format(data['parse_div'].size()),"dtype: ", data['parse_div'].dtype,torch.max(data['parse_div']),torch.min(data['parse_div']))
                print("Mdiv:{}".format(data['misalign_mask'].size()),"dtype: ", data['misalign_mask'].dtype,torch.max(data['misalign_mask']),torch.min(data['misalign_mask']))
                print("gt:{}".format(data['ground_truth_image'].size()),"dtype: ", data['ground_truth_image'].dtype,torch.max(data['ground_truth_image']),torch.min(data['ground_truth_image']))


            Ia = Variable(data['img_agnostic'])
            P  = Variable(data['pose'])
            Wc = Variable(data['warped_c'])
            Ma = Variable(data['agnostic_mask'])
            S =  Variable(data['parse'])
            Sdiv = Variable(data['parse_div'])
            Mdiv = Variable(data['misalign_mask'])
            Igt  = Variable(data['ground_truth_image'])
            paths = data['path']

            (losses, generated),vgg_loss_map_list,Feat_map_list = model(Ia, P, Wc, S, Sdiv, Mdiv,Ma ,Igt, infer=save_fake)
        #print(torch.unique(generated))
        # sum per device losses

        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_dict = dict(zip(model.module.loss_names, losses))
        # calculate final loss scalar
        if epoch>opt.niter_stable:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat',0) + loss_dict.get('G_VGG',0) + loss_dict.get('G_L1', 0) * 0
        else:
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) *0
            loss_G = loss_dict['G_GAN'] * 0 + loss_dict.get('G_GAN_Feat',0) * 0 + loss_dict.get('G_VGG',0) + loss_dict.get('G_L1', 0)
        ############### Backward Pass ####################
        # update generator weights
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
        '''errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
        eta = (time.time() - epoch_start_time) * (len(dataset) / opt.batchSize - i) / (i - save_epoch_iter + 1)
        visualizer.print_current_errors(epoch, epoch_iter, errors, eta)'''
        ############## Display results and errors ##########
        ### print out errors

        if total_steps % opt.print_freq == print_delta:
            errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
            eta = (time.time() - epoch_start_time)* (len(dataset)/opt.batchSize - i)/(i - save_epoch_iter +1)
            visualizer.print_current_errors(epoch, epoch_iter, errors, eta)
            visualizer.plot_current_errors(errors, total_steps)
            #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])

            ### display output images
            if True:
                for data_index in data['index'].tolist():
                    if not once2: # debugging inputs
                        # to check result only 
                        visuals = OrderedDict([('gen{:05d}'.format(data_index), util.tensor2im(generated.data[0])),
                                   ('gt{:05d}'.format(data_index), util.tensor2im(data['ground_truth_image'][0]))])
                    else:
                        if opt.inpaint:
                            visuals = OrderedDict([('gen{:05d}'.format(data_index), util.tensor2im(generated.data[0])),
                                                   ('Ia{:05d}'.format(data_index),
                                                    util.tensor2im(data['img_agnostic'][0])),
                                                   ('warpedC{:05d}'.format(data_index),
                                                    util.tensor2im(data['warped_c'][0])),
                                                   ('pose{:05d}'.format(data_index), util.tensor2im(data['pose'][0])),
                                                   ('Sdiv{:05d}'.format(data_index),
                                                    util.tensor2label(data['parse_div'][0], 8)),
                                                   ('Mdiv{:05d}'.format(data_index),
                                                    util.tensor2im(data['misalign_mask'][0])),
                                                   ('gt{:05d}'.format(data_index),
                                                    util.tensor2im(data['ground_truth_image'][0])),
                                                   # ('cm{:05d}'.format(data_index), util.tensor2im(c_mask[0])),
                                                   # ('gt2{:05d}'.format(data_index), util.tensor2im(Igt2[0])),

                                                   ])
                        else:
                            visuals = OrderedDict([('gen{:05d}'.format(data_index), util.tensor2im(generated.data[0])),
                                       ('Ia{:05d}'.format(data_index), util.tensor2im(data['img_agnostic'][0])),
                                       ('warpedC{:05d}'.format(data_index), util.tensor2im(data['warped_c'][0])),
                                       ('S{:05d}'.format(data_index), util.tensor2label(data['parse'][0], 7)),
                                       ('pose{:05d}'.format(data_index), util.tensor2im(data['pose'][0])),
                                       ('Sdiv{:05d}'.format(data_index), util.tensor2label(data['parse_div'][0], 8)),
                                       ('Mdiv{:05d}'.format(data_index), util.tensor2im(data['misalign_mask'][0])),
                                       ('gt{:05d}'.format(data_index), util.tensor2im(data['ground_truth_image'][0])),
                                       #('cm{:05d}'.format(data_index), util.tensor2im(c_mask[0])),
                                       #('gt2{:05d}'.format(data_index), util.tensor2im(Igt2[0])),

                                       ])
                        #once2 = False
                '''
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                   ('synthesized_image', util.tensor2im(generated.data[0])),
                                   ('real_image', util.tensor2im(data['image'][0]))])
                                   
                Ia = Variable(data['img_agnostic'])
                P  = Variable(data['pose'])
                Wc = Variable(data['warped_c'])
                Ma = Variable(data['agnostic_mask'])
                S =  Variable(data['parse'])
                Sdiv = Variable(data['parse_div'])
                Mdiv = Variable(data['misalign_mask'])
                Igt  = Variable(data['ground_truth_image'])
                '''
                if True:
                    #visualizer.save_current_results(visuals, epoch, total_steps)
                    for b in range(len(paths)):
                        name = paths[b].split('/')[-1].split('.')[0]
                        img_dir = os.path.join(opt.checkpoints_dir,opt.name,'web', 'images','epoch%.3d_step%.5d_%s_%s.jpg' % (epoch,total_steps, 'vgg_loss_map', name))
                        plt.figure(figsize=(16, 8))
                        for i in range(5):
                            loss_map = np.squeeze(vgg_loss_map_list[i][b].cpu().numpy())
                            plt.subplot(2, 4, i + 1), plt.imshow(loss_map), \
                            plt.title(
                                    f"%dth feature\navr:%.5f"%(i,np.average(loss_map))),
                            plt.colorbar(),
                            plt.axis('off')
                        fake_img = np.transpose(((generated.detach().cpu().numpy()[b]+1)/2*255).astype(np.uint8),(1,2,0))
                        true_img = np.transpose(((Igt.detach().cpu().numpy()[b]+1)/2*255).astype(np.uint8),(1,2,0))
                        plt.subplot(2,4,7),plt.imshow(fake_img),plt.title('fake'),plt.axis('off')
                        plt.subplot(2,4,8),plt.imshow(true_img),plt.title('true'),plt.axis('off')
                        plt.savefig(img_dir)
                        plt.clf()

                        img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web', 'images',
                                               'epoch%.3d_step%.5d_%s_%s.jpg' % (epoch,total_steps, 'Feat_loss_map', name))
                        for i in range(len(Feat_map_list)):
                            for j in range(len(Feat_map_list[i])):
                                loss_map = np.squeeze(Feat_map_list[i][j].cpu().numpy())
                                plt.subplot(2,5,5*i+j+1), plt.imshow(loss_map),\
                                plt.title(f"%dth D, %dth feature\navr:%.5f"%(i,j,np.average(loss_map))),
                                plt.colorbar(),
                                plt.axis('off')
                        plt.subplot(2,5,5),plt.imshow(fake_img),plt.title('fake'),plt.axis('off')
                        plt.subplot(2, 5, 10), plt.imshow(true_img), plt.title('true'),plt.axis('off')
                        plt.savefig(img_dir)
                        plt.clf()




                if False:
                    visualizer.display_current_results(visuals, epoch, total_steps)

        ### save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.module.save('latest')            
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break
       
    # end of epoch
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter+ opt.niter_stable + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))        
        model.module.save('latest')
        model.module.save(epoch)
        np.savetxt(iter_path, (epoch+1, 0), delimiter=',', fmt='%d')

    ### linearly decay learning rate after certain iterations
    if epoch > opt.niter + opt.niter_stable:
        model.module.update_learning_rate()
    #call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])


