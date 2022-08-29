# Beta 0.5 conv-64
for i in `seq 20`
do
  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0.5 --net conv64 --lr_base 0.001 --Novel_Mosaic_rate 1 --k_shot 1 --mosaic_num 200 --id 1503 --gpu 0 --num_sim 2 --idx_task $i --mosaic_ori_num 1 --proto_loss
done