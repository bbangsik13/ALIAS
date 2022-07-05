import torch

def create_model(opt):

    from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
    if opt.isTrain:
        model = Pix2PixHDModel()
    else:
        model = InferenceModel()
    model.initialize(opt)

    print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
