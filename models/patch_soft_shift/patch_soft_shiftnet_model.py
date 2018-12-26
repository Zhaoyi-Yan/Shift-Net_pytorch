from models.shift_net.shiftnet_model import ShiftNetModel


class PatchSoftShiftNetModel(ShiftNetModel):
    def name(self):
        return 'PatchSoftShiftNetModel'
