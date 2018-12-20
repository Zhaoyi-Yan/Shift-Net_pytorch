from models.accelerated_shift_net.accelerated_shiftnet_model import ShiftNetModel


class SoftShiftNetModel(ShiftNetModel):
    def name(self):
        return 'SoftShiftNetModel'
