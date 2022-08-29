#4.28 减少了epoch数量，同时减少了lr
#resnet12  0.78
#for i in `seq 20`
#do
#  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0.5 --net resnet12 --lr_base 0.001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 1521 --gpu 2 --idx_task $i --proto_loss --num_sim 2 --mosaic_ori_num 2
#done


# ce loss
for i in `seq 10`
do
  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0.5 --net resnet18 --lr_base 0.0005 --Novel_Mosaic_rate 0.8 --k_shot 5 --mosaic_num 500 --id 1525 --gpu 2  --idx_task $i --num_sim 5 --mosaic_ori_num 0
done

#for i in `seq 20`
#do
#  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0.5 --net resnet18 --lr_base 0.001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 1521 --gpu 2 --idx_task $i --proto_loss --num_sim 3 --mosaic_ori_num 0
#done