
# lr_base = 0.0002   lr_backbone = 0.0001

#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0005 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 100 --id 1520 --gpu 2 --num_sim 3 --mosaic_ori_num 3 --num_task 100  --Epoch 30 --grid 3  --fsl_val  --base_mos --yaml fsl_train_base_cls.yaml

#t0:
# 取消原始图片
python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.001 --Novel_Mosaic_rate 0.8 --k_shot 5 --mosaic_num 100 --id 152 --gpu 2 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 30 --grid 2  --fsl_val  --base_mos --yaml fsl_train_base_cls.yaml --mosaic_center --random_mosaic

#f3:
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 400 --id 153 --gpu 3 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 30 --grid 2  --fsl_val  --base_mos --yaml fsl_train_base_cls.yaml  --random_mosaic

#t0:   去掉了原来的颜色变换，尽量使用原图进行简单的mosaic   选取为最近的100个图片   分开的优化器   学习率为0.001  0.0001，  10epoch（4，7  0.4decay） 见151日志 (不少任务过于差了)  ： 0.781 ==> 更大的transforms 尝试更大的backbone lr
#f3  ： old setting  400mosaic  30epoch  相同的优化器，同时进行lr decay（0.4） 见日志152