#for i in {1,2,3,4,5,6,7,8,9,10}
#do
#python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 0.9 --k_shot 1 --mosaic_num 200 --id 0
#done
#
#for i in {1,2,3,4,5,6,7,8,9,10}
#do
#python fsl_train.py --sim_way 2 --dist_scale 1 --Novel_Mosaic_rate 0.9 --k_shot 5 --mosaic_num 200 --id 0
#done

#python fsl_train_old.py --sim_way 2 --dist_scale 0.7 --Novel_Mosaic_rate 1

for i in `seq 20`
do
  echo $i-1
done