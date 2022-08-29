## 目前实验：在test阶段  100task Acc 只有 0.7577

## 2022/7/2:
    在Test 阶段 进行100task的 ProtoNet 实验：   gpu: 1  10 epoch
        结果:    
            (1)  0.7851
            (2)     no back decay : 
     
    在Val 阶段  进行 100 task 的 idea 实验:     gpu: 2  50 epoch
        修改点：
                (1)8/16  : 0.7854
                (2)8/16  no back decay:  0.7791
                (3)8/16                  20epoch:   0.7821
        结果:

是否可以尝试网格化探索超参数：  lr: 0.001--0.0001 、 mosaic 数量 、 7


8/7:
1、test_mosaic :(在test阶段引入prompt思想)
lr ：0.0004  与没有引入的效果相近   
 没有必要太多的epoch，同时提高test阶段的mosaic，也许能使结果更为稳定。

2、能不能修改mosaic方式，来提升运行速度。（节约时间，使得mosaic个数能大幅提高）

  构建方法：    finished
  a、removeDir：清空临时文件夹
  b、按照sample次序保存文件夹,用一个字典来保存，然后在__getitem()__阶段直接读取
  
3、在单独去除support阶段的mosaic和support 的location pretext后，Acc下降到0.76