def create_model(opt):
    model = None
    if opt.model == 'shiftnet':
        assert(opt.dataset_mode == 'aligned')
        from models.shift_net.shiftnet_model import ShiftNetModel
        model = ShiftNetModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

