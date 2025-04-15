import cv2
import numpy as np
import matplotlib.pyplot as plt

#  更改相对路径开始使用

def extract_center_points(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 滤波降噪
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 二值化
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # 提取激光线中心
    height, width = binary.shape
    center_points = []
    for x in range(width):
        column = binary[:, x]
        y = np.argmax(column)  # 找到最亮点
        if column[y] > 0:  # 确保有亮点
            center_points.append((x, y))

    return np.array(center_points), image

def fit_curve(points):
    # 多项式拟合
    x_data, y_data = points[:, 0], points[:, 1]
    poly_coeffs = np.polyfit(x_data, y_data, deg=2)  # 二次拟合
    return np.poly1d(poly_coeffs)

def compute_similarity(curve1, curve2, x_data):
    # 计算两条曲线在相同x范围内的误差，忽略常数项
    y1 = curve1(x_data) - curve1.c[-1]  # 减去常数项
    y2 = curve2(x_data) - curve2.c[-1]  # 减去常数项
    mse = np.mean((y1 - y2) ** 2)
    return mse

# 提取两段光线的中心点
points_left, image_left = extract_center_points("./ok_compressed2.jpg")
points_right, image_right = extract_center_points("./ng_compressed2.jpg")

# 拟合曲线
curve_left = fit_curve(points_left)
curve_right = fit_curve(points_right)

# 计算匹配误差
x_common = np.linspace(0, min(points_left[-1, 0], points_right[-1, 0]), num=500)
similarity_score = compute_similarity(curve_left, curve_right, x_common)
print("光线匹配误差：", similarity_score)

# 在原图中显示拟合曲线
plt.figure(figsize=(12, 8))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_left, cv2.COLOR_GRAY2RGB))
plt.plot(x_common, curve_left(x_common), 'r-', label='Left Fitted Curve')
plt.scatter(points_left[:, 0], points_left[:, 1], color='blue', s=1, label='Left Points')
plt.title("Left Image with Fitted Curve")
plt.legend()
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_right, cv2.COLOR_GRAY2RGB))
plt.plot(x_common, curve_right(x_common), 'y-', label='Right Fitted Curve')
plt.scatter(points_right[:, 0], points_right[:, 1], color='green', s=1, label='Right Points')
plt.title("Right Image with Fitted Curve")
plt.legend()
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
