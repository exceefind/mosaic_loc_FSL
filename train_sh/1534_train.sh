# Beta 0.5 conv-64
for i in `seq 20`
do
  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0.5 --net resnet12 --lr_base 0.001 --Novel_Mosaic_rate 1 --k_shot 1 --mosaic_num 200 --id 1534 --gpu 3 --num_sim 2 --idx_task $i --mosaic_ori_num 0 --proto_loss
done