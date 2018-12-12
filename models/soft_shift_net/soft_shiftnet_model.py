from models.shift_net.shiftnet_model import ShiftNetModel


class SoftShiftNetModel(ShiftNetModel):
    def name(self):
        return 'SoftShiftNetModel'
