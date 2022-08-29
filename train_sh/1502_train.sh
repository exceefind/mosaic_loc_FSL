for i in `seq 20`
do
  python fsl_train.py --sim_way 2 --dist_scale 1 --beta 0 --net conv64 --lr_base 0.01 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 1502 --gpu 0 --have_Original --idx_task $i
done