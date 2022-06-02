
def CreateDataLoader(opt):
    if opt.use_warped:
        from data.custom_dataset_data_loader import CustomDatasetDataLoader
        data_loader = CustomDatasetDataLoader()
        data_loader.initialize(opt)
    else:
        from data.Style_Viton_DataLoader import VITONDataLoader,VITONDataset
        test_dataset = VITONDataset(opt)
        data_loader = VITONDataLoader(opt, test_dataset)

    print(data_loader.name())
    return data_loader
