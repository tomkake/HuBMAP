import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import directed_hausdorff

# setAとsetBの座標
setA = np.array([[0, 0], [1, 1], [2, 2]])
setB = np.array([[0, 1], [1, 3], [2, 3]])


def hausdorff_distance(setA, setB):
    hd1 = directed_hausdorff(setA, setB)[0]
    hd2 = directed_hausdorff(setB, setA)[0]

    return max(hd1, hd2)


# Hausdorff距離の計算
hd = hausdorff_distance(setA, setB)

# matplotlibで描画
plt.figure()
plt.scatter(setA[:, 0], setA[:, 1], color="red", label="setA")  # setAを赤色で描画
plt.scatter(setB[:, 0], setB[:, 1], color="blue", label="setB")  # setBを青色で描画
plt.title(f"Hausdorff Distance: {hd}")  # タイトルとしてHausdorff距離を表示
plt.legend()  # 凡例を表示
plt.savefig("figure01.jpg")
