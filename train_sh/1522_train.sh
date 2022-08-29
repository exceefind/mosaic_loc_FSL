# 减少了epoch数量，同时减少了lr
#ce loss sim_way 0 lr 0.005  0.78
#
#for i in `seq 20`
#do
#  python fsl_train.py --sim_way 0 --dist_scale 1 --beta 0.5 --net resnet12 --lr_base 0.005 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 400 --id 1522 --gpu 1  --idx_task $i --num_sim 2 --mosaic_ori_num 0
#done


for i in `seq 10`
do
  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 500 --id 1522 --gpu 1  --idx_task $i --num_sim 5 --mosaic_ori_num 0
done

#下一步考虑，回到mosaic_ori 为0，从头开始训练30-50个epoch

#for i in `seq 10`
#do
#  python fsl_train.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 500 --id 1522 --gpu 1  --idx_task $i --num_sim 3 --mosaic_ori_num 0
#done