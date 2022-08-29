
# proto loss  sota settting
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1502 --gpu 0 --num_sim 5 --mosaic_ori_num 3 --fsl_val --num_task 50  --Epoch 40 --grid 3 --proto_loss --yaml fsl_sota.yaml --train_head

#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 1 --net resnet12 --lr_base 0.0002 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1500 --gpu 0 --num_sim 5 --mosaic_ori_num 3 --fsl_val --num_task 50  --Epoch 150 --grid 3

#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 1 --net resnet12 --lr_base 0.0002 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1502 --gpu 1 --num_sim 5 --mosaic_ori_num 1 --fsl_val --num_task 50  --Epoch 150 --grid 2 --random_mosaic --yaml fsl_sota.yaml

# proto loss  0.7843
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.6 --net resnet12 --lr_base 0.0004 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 5 --id 1512 --gpu 2 --num_sim 5 --mosaic_ori_num 3 --fsl_val --num_task 100  --Epoch 100 --grid 3  --fsl_val --proto_loss

# test：
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.6 --net resnet12 --lr_base 0.0002 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1500 --gpu 0 --num_sim 5 --mosaic_ori_num 3 --num_task 100  --Epoch 100 --grid 3  --proto_loss


#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.6 --net resnet12 --lr_base 0.0002 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1510 --gpu 1 --num_sim 5 --mosaic_ori_num 1 --fsl_val --num_task 50  --Epoch 150 --grid 2 --proto_loss


#  2022/8/4
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.6 --net resnet12 --lr_base 0.0004 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 10 --id 1510 --gpu 0 --num_sim 5 --mosaic_ori_num 3 --fsl_val --num_task 100  --Epoch 50 --grid 3  --fsl_val --proto_loss



# 纯粹的 proto net  beta=0,  lr = 0.0005,  mosaic_num  = 1,  not test_mos
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0005 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1511 --gpu 2 --num_sim 5 --mosaic_ori_num 5 --fsl_val --num_task 100  --Epoch 50 --grid 3  --fsl_val --proto_loss

# 在test中引入mosaic:
#  f2
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.1 --net resnet12 --lr_base 0.00008 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 2 --id 1503 --gpu 1 --num_sim 5 --mosaic_ori_num 5 --fsl_val --num_task 100  --Epoch 30 --grid 3  --fsl_val --proto_loss

# f1
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.2 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 2 --id 1503 --gpu 2 --num_sim 5 --mosaic_ori_num 3 --fsl_val --num_task 100  --Epoch 40 --grid 3  --proto_loss

#小幅mosaic和protoNet尝试：    380task  acc  ： 0.7759
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.2 --net resnet12 --lr_base 0.0005 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 2 --id 1523 --gpu 2 --num_sim 5 --mosaic_ori_num 5 --num_task 1000  --Epoch 10 --grid 3  --proto_loss


#   base_mosaic  0.1 decay  0.0005  :  0.7502 (100 task)
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.0005 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 20 --id 1571 --gpu 7 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 20 --grid 2  --fsl_val  --base_mos

# base_mosaic  0.1 decay  0.0002   lr 分离  :  0.7405
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.0002 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 50 --id 1571 --gpu 7 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 30 --grid 2  --fsl_val  --base_mos

# t0  base_mosaic    lr 分离   back :0.00005     lr_base 0.0001  sim_num=5   :0.7514
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 50 --id 1515 --gpu 1 --num_sim 5 --mosaic_ori_num 1 --num_task 100  --Epoch 30 --grid 2  --fsl_val  --base_mos

# f2  base_mosaic    lr 分离   back :0.0001     lr_base 0.001  lr_back 0.0001  sim_num=5  temperature = 4  epoch =50
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0008 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 10 --id 1565 --gpu 6 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 50 --grid 2  --fsl_val  --base_mos  --yaml fsl_train_base_cls.yaml --random_mosaic

#     lr 分离   back :0.00005     lr_base 0.0001  sim_num=5  temperature = 4  epoch =15  :  0.7894(val)  100 task
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 2 --id 1572 --gpu 7 --num_sim 5 --mosaic_ori_num 5 --num_task 100  --Epoch 15 --grid 3  --fsl_val  --proto_loss

# f1: 纯粹的proto  :  acc= 0.7889    事实上与小幅的support mosaic的性能机会一致==>  改为base_mos 10  开局即是巅峰？ acc=0.7914 ==> lr_back = 8e-4
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 1 --id 1572 --gpu 7 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 50 --grid 2  --fsl_val  --proto_loss

#t0:  base_mos 100   epoch=20   lr_backbone = 0.0001 = lr *0.5  not random_mos 3/9
python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 100 --id 1510 --gpu 1 --num_sim 3 --mosaic_ori_num 4 --num_task 100  --Epoch 30 --grid 3  --fsl_val  --base_mos --proto_loss --mosaic_center


# base_mosaic  0.1 decay  0.001   lr 分离  :  0.7365   学习率过大
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 20 --id 1543 --gpu 4 --num_sim 3 --mosaic_ori_num 1 --num_task 100  --Epoch 25 --grid 2  --fsl_val  --base_mos


# base_mosaic  no decay  :0.7349
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0.4 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 20 --id 1521 --gpu 2 --num_sim 5 --mosaic_ori_num 1 --num_task 100  --Epoch 20 --grid 2  --fsl_val  --base_mos


#  old  扩大 mosaic _num
#python fsl_train_copy.py --sim_way 1 --dist_scale 1 --beta 0 --net resnet12 --lr_base 0.0005 --Novel_Mosaic_rate 1 --k_shot 20 --mosaic_num 1 --id 1512 --gpu 1 --num_sim 5 --mosaic_ori_num 5 --num_task 300  --Epoch 10 --grid 3  --proto_loss
