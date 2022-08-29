#for i in `seq 10`
#do
#  python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#done

#python fsl_train.py --sim_way 2 --dist_scale 1 --beta 0.001 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#
#python fsl_train.py --sim_way 2 --dist_scale 1 --beta 0.1 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#
#python fsl_train.py --sim_way 2 --dist_scale 1 --beta 0.4 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#
#python fsl_train.py --sim_way 0 --dist_scale 1 --beta 0.001 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#
#python fsl_train.py --sim_way 0 --dist_scale 1 --beta 0.1 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500


#python fsl_train.py --sim_way 2 --dist_scale 1 --beta 1 --lr_base 0.05 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 100 --id 1500  --have_Original

#python fsl_train.py --sim_way 2 --dist_scale 0.5 --beta 1 --lr_base 0.001 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500
#
#python fsl_train.py --sim_way 2 --dist_scale 0.5 --beta 1 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500

#python fsl_train.py --sim_way 0 --dist_scale 1 --beta 0 --lr_base 0.005 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1500  --have_Original


#验证不同学习率和beta的影响
python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0.5

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0.7

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0.5

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0.7

#验证不同学习率和beta， mosaic 2个pach的性能
python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0 --mosaic_ori_num 2

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0.5 --mosaic_ori_num 2

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.005 --beta 0.7 --mosaic_ori_num 2

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0 --mosaic_ori_num 2

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0.5 --mosaic_ori_num 2

python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 200 --id 152 --net conv64 --gpu 0 --lr_base 0.01 --beta 0.7 --mosaic_ori_num 2
