from create_data import *
from drcnn import *
from modulation import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import time
from keras import backend as K
K.set_image_dim_ordering('th')

single_sample_num = 4000  # 一个调制方式用多少个样本
sample_long = 128  # 每个样本多长
output_num = 4  # 模型输出大小，识别调制方式数量
fs = 200  # 采样率
fc = 4  # 载波频率
SNR = 5  # 信噪比
train_rate = 0.8  # 训练集比例

# 生成数据
x, y = generate_data(fs, fc, SNR, single_sample_num, sample_long)

# 生成索引，以达到打乱的目的
cols = x.shape[0]
index = np.arange(cols)
np.random.shuffle(index)
# 对y进行One-Hot编码
y = to_categorical(y)

# 分离训练集测试集
train_split = int(cols * train_rate)
x_train, x_test = x[index[:train_split]], x[index[train_split:]]
y_train, y_test = y[index[:train_split]], y[index[train_split:]]

# 建立模型
model = build_drcnn(sample_long, output_num)
# 设置回调函数
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto')
tensorb = TensorBoard(log_dir='./borad', write_grads=True, write_images=True)

history = model.fit(x_train,
                    y_train,
                    batch_size=32,
                    epochs=70,
                    validation_split=0.2,
                    callbacks=[reduce_lr, tensorb],
                    verbose=1
                    )

# 绘制训练验证的准确率
his = history.history
plt.plot(his['acc'])
plt.plot(his['val_acc'])
plt.title('Model accuracy SNR='+str(SNR))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
fig_name = './model_acc_SNR' + str(SNR) + '_' + time.strftime('%m-%d_%H-%M', time.localtime()) + '.jpg'
plt.savefig(fig_name)

# 绘制训练验证的损失
his = history.history
plt.plot(his['loss'])
plt.plot(his['val_loss'])
plt.title('Model loss SNR='+str(SNR))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
fig_name = './model_loss_SNR' + str(SNR) + '_' + time.strftime('%m-%d_%H-%M', time.localtime()) + '.jpg'
plt.savefig(fig_name)

# 画混淆矩阵
y_test_pred = model.predict(x_test)
none_cate_y_test = list(map(np.argmax, y_test))
none_cate_y_test_pred = list(map(np.argmax, y_test_pred))
c = confusion_matrix(none_cate_y_test, none_cate_y_test_pred)
fig, ax = plt.subplots()
sns.heatmap(c, annot=True, ax=ax)
ax.set_title('confusion matrix snr='+str(SNR))
ax.set_xlabel('Predict')
ax.set_ylabel('True')
fig_name = './ConfusionMatrix_SNR' + str(SNR) + '_' + time.strftime('%m-%d_%H-%M', time.localtime()) + '.jpg'
fig.savefig(fig_name)

# 计算准确率
test_acc = accuracy_score(none_cate_y_test, none_cate_y_test_pred)
print('test accuracy:')
print(test_acc)
