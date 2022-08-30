#image label :

#not base data
#python center_sample_extract.py --gpu 2 --net resnet12 --model_continue 1 --no_train

#首先退化成 protoNet  测试其基本性能

python fsl_ImageLabel.py --sim_way 1 --dist_scale 1 --alpha 0.1 --beta 1 --net resnet12 --lr_base 0.0001 --Novel_Mosaic_rate 1 --k_shot 5 --mosaic_num 100 --id 153 --gpu 3 --num_sim 3 --mosaic_ori_num 100 --num_task 100  --Epoch 20 --grid 2  --fsl_val  --base_mos --yaml fsl_Img_Label.yaml --mosaic_center --random_mosaic --test_mosaic
