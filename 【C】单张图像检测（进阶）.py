#!/usr/bin/env python
# coding: utf-8

# B站：同济子豪兄（https://space.bilibili.com/1900783）
# 
# 微信公众号：人工智能小技巧
# 
# 张子豪 2021-07-12

# # 导入工具包

# In[26]:


# opencv-python
import cv2

# mediapipe人工智能工具包
import mediapipe as mp

# 进度条库
from tqdm import tqdm

# 时间库
import time

# 导入python绘图matplotlib
import matplotlib.pyplot as plt
# 使用ipython的魔法方法，将绘制出的图像直接嵌入在notebook单元格中
get_ipython().run_line_magic('matplotlib', 'inline')


# In[27]:


# 定义可视化图像函数
def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# # 导入手部关键点检测模型

# In[28]:


# 导入solution
mp_hands = mp.solutions.hands

# 导入模型
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=2,                # 最多检测几只手
                       min_detection_confidence=0.5,   # 置信度阈值，过滤低于该阈值的预测结果
                       min_tracking_confidence=0.5)    # 追踪阈值

# 导入绘图函数
mpDraw = mp.solutions.drawing_utils 


# # 读入图像，输入模型，获取预测结果

# In[29]:


# 从图片文件读入图像，opencv读入为BGR格式
img = cv2.imread('./images/camera1.jpg')

# 水平镜像翻转图像，使图中左右手与真实左右手对应
# 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
img = cv2.flip(img, 1)

# BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 将RGB图像输入模型，获取预测结果
results = hands.process(img_RGB)


# In[30]:


look_img(img)


# # 分析模型输出结果

# # 置信度和左右手

# In[31]:


results.multi_handedness


# 索引为0的手

# In[32]:


results.multi_handedness[0].classification[0].score


# In[33]:


results.multi_handedness[0].classification[0].label


# 索引为1的手

# In[34]:


results.multi_handedness[1].classification[0].score


# In[35]:


results.multi_handedness[1].classification[0].label


# # 关键点坐标

# In[36]:


# 预测出手的个数
len(results.multi_hand_landmarks)


# In[37]:


# 所有手的所有关键点坐标
results.multi_hand_landmarks


# In[38]:


# 获取索引为0的手的关键点坐标
results.multi_hand_landmarks[0]


# In[39]:


# 获取索引为1的手的关键点坐标
results.multi_hand_landmarks[1]


# 索引为1的手的第20号关键点的坐标

# In[40]:


results.multi_hand_landmarks[1].landmark[20]


# In[41]:


results.multi_hand_landmarks[1].landmark[20].x


# In[42]:


# 获取图像宽高
h, w = img.shape[0], img.shape[1]


# In[43]:


h


# In[44]:


w


# In[45]:


results.multi_hand_landmarks[1].landmark[20].x * w


# In[46]:


results.multi_hand_landmarks[1].landmark[20].y * h


# ## 关键点之间的连接关系（骨架）

# In[47]:


mp_hands.HAND_CONNECTIONS


# # 可视化检测结果

# In[48]:


if results.multi_hand_landmarks: # 如果有检测到手
    # 遍历每一只检测出的手
    for hand_idx in range(len(results.multi_hand_landmarks)):
        hand_21 = results.multi_hand_landmarks[hand_idx] # 获取该手的所有关键点坐标
        mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS) # 可视化


# In[49]:


look_img(img)


# # 整理代码

# In[50]:


# opencv-python
import cv2

# mediapipe人工智能工具包
import mediapipe as mp

# 进度条库
from tqdm import tqdm

# 时间库
import time

# 导入python绘图matplotlib
import matplotlib.pyplot as plt
# 使用ipython的魔法方法，将绘制出的图像直接嵌入在notebook单元格中
get_ipython().run_line_magic('matplotlib', 'inline')

# 定义可视化图像函数
def look_img(img):
    '''opencv读入图像格式为BGR，matplotlib可视化格式为RGB，因此需将BGR转RGB'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()


# 导入solution
mp_hands = mp.solutions.hands
# 导入模型
hands = mp_hands.Hands(static_image_mode=False,        # 是静态图片还是连续视频帧
                       max_num_hands=4,                # 最多检测几只手
                       min_detection_confidence=0.5,   # 置信度阈值，过滤低于该阈值的预测结果
                       min_tracking_confidence=0.5)    # 追踪阈值
# 导入绘图函数
mpDraw = mp.solutions.drawing_utils 

# 从图片文件读入图像，opencv读入为BGR格式
img = cv2.imread('./images/camera1.jpg')

# 水平镜像翻转图像，使图中左右手与真实左右手对应
# 参数 1：水平翻转，0：竖直翻转，-1：水平和竖直都翻转
img = cv2.flip(img, 1)
# BGR转RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 将RGB图像输入模型，获取预测结果
results = hands.process(img_RGB)
if results.multi_hand_landmarks: # 如果有检测到手
    # 遍历每一只检测出的手
    for hand_idx in range(len(results.multi_hand_landmarks)):
        hand_21 = results.multi_hand_landmarks[hand_idx] # 获取该手的所有关键点坐标
        mpDraw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS) # 可视化

look_img(img)

cv2.imwrite('C.jpg',img)


# In[ ]:




