import matplotlib.pyplot as plt
import numpy as np

def fuzzy_rule_2(dif):

    x=0 #LOW
    y=0 #MEDIUM
    z=0 #HIGH
    if dif<-10:
        x=1
    elif dif>=-10 and dif<=0:
        x=(-dif)/10
    
    if dif>0 and dif<=15:
        y=(15-dif)/15
    elif dif<=0 and dif >=-15:
        y=(dif+15)/15

    if dif>=10:
        z=1
    elif dif>=0 and dif<10:
        z=(dif)/10

    return [x,y,z]

# 生成 0 到 50 范围内的输入值
inputs = np.arange(-40, 41, 1)
outputs = np.array([fuzzy_rule_2(x) for x in inputs])

# 画出 X，Y，Z 随着前方距离的变化的图像
plt.plot(inputs, outputs[:,0], label='X (LOW)')
plt.plot(inputs, outputs[:,1], label='Y (MEDIUM)')
plt.plot(inputs, outputs[:,2], label='Z (HIGH)')
plt.title('Fuzzy Rule 2: X, Y, Z vs Difference Between Right Distance and Left Distance')
plt.xlabel('Difference Between Right Distance and Left Distance')
plt.ylabel('Fuzzy Rule 2 Output')
plt.legend()
plt.grid(True)
plt.show()