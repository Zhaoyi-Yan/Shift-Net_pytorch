#!/bin/bash
name='re_gan_0.2'
dataroot='./datasets/Paris/test' # to be
which_model_netG='unet_shift_triple'
which_model_netD='basic'
model='shiftnet'
gan_weight=0.5
skip=0
mask_type='center'

gpu_ids='0'

start_idx=2

final_epoch=70

step=2
count=0

j=$start_idx

while [ $j -le $final_epoch ]; do
{
j=$(($count * $step + $start_idx))
echo "Testing epoch : "${j}

python test.py --gpu_ids=${gpu_ids} --dataroot=${dataroot} --which_model_netG=${which_model_netG} --model=${model} --name=${name} --which_epoch=${j} --gan_weight=${gan_weight} --skip=${skip} --mask_type=${mask_type} --which_model_netD=${which_model_netD}
let count++
}
done

