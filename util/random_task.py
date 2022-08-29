import random

import numpy as np

task_num = 20
random_task = []
for i in range(task_num):
    task = random.sample(range(20),5)
    random_task.append(task)

np.save("../random_task.npy",random_task)

random_task = np.load("../random_task.npy")
