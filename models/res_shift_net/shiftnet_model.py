from models.shift_net.shiftnet_model import ShiftNetModel


class ResShiftNetModel(ShiftNetModel):
    def name(self):
        return 'ResShiftNetModel'