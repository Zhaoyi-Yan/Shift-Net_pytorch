from models.shift_net.shiftnet_model import ShiftNetModel


class ResPatchSoftShiftNetModel(ShiftNetModel):
    def name(self):
        return 'ResPatchSoftShiftNetModel'
