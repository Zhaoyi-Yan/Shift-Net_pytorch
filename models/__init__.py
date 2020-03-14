def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'shiftnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.shift_net.shiftnet_model import ShiftNetModel
        model = ShiftNetModel()

    elif opt.model == 'res_shiftnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.res_shift_net.shiftnet_model import ResShiftNetModel
        model = ResShiftNetModel()

    elif opt.model == 'patch_soft_shiftnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.patch_soft_shift.patch_soft_shiftnet_model import PatchSoftShiftNetModel
        model = PatchSoftShiftNetModel()

    elif opt.model == 'res_patch_soft_shiftnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.res_patch_soft_shift.res_patch_soft_shiftnet_model import ResPatchSoftShiftNetModel
        model = ResPatchSoftShiftNetModel()

    elif opt.model == 'face_shiftnet':
        assert (opt.dataset_mode == 'aligned' or opt.dataset_mode == 'aligned_resized')
        from models.face_shift_net.face_shiftnet_model import FaceShiftNetModel
        model = FaceShiftNetModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
