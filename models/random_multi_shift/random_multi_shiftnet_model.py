from models.shift_net.shiftnet_model import ShiftNetModel


class RandomMultiShiftNetModel(ShiftNetModel):
    def name(self):
        return 'RandomMultiShiftNetModel'
