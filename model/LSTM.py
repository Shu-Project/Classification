import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# 1. 数据加载和预处理 (保留不变)
def load_data():
    data = []
    labels = []

    for move_num in range(1, 7):
        move_file = f"../Datas/Move{move_num}.xlsx"
        df = pd.read_excel(move_file)

        # 假设 'Time' 列是时间戳，其他列是信道数据
        time_series = df['Time'].values
        channels = df.drop(columns=['Time']).values

        # 使用滑动窗口进行切分
        for start in range(0, len(time_series) - 80, 40):
            window = channels[start:start + 80]

            # 提取统计特征
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            max_val = np.max(window, axis=0)
            min_val = np.min(window, axis=0)
            skewness = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 3), axis=0, arr=window)
            kurtosis = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 4), axis=0, arr=window)

            # 将原始信号数据和统计特征拼接
            feature_vector = np.concatenate([mean, std, max_val, min_val, skewness, kurtosis])

            data.append(feature_vector)
            labels.append(move_num - 1)

    return np.array(data), np.array(labels)


# 加载数据
data, labels = load_data()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 训练数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 测试数据加载器
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. 改进的 STML 模型（卷积层 + 双向 LSTM）
class STMLModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(STMLModel, self).__init__()

        # 1. 空间特征提取部分 (CNN)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)  # 卷积核尺寸适当减小
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)  # 避免卷积尺寸过小

        self.pool = nn.MaxPool1d(2)  # 保留池化层，但减少池化的次数
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.batch_norm2 = nn.BatchNorm1d(128)

        # 2. 时间序列特征提取部分 (双向 LSTM)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        # 3. 全连接层 (分类器)
        self.fc1 = nn.Linear(256 * 2, hidden_size)  # 双向LSTM，维度翻倍
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.unsqueeze(1)

        # 空间特征提取 (卷积层)
        x = self.batch_norm1(self.relu(self.conv1(x)))
        x = self.pool(x)

        x = self.batch_norm2(self.relu(self.conv2(x)))
        x = self.pool(x)

        # 将conv层的输出平坦化，以输入LSTM
        x = x.permute(0, 2, 1)

        # 时间序列特征提取 (LSTM)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # 取LSTM的最后一个时间步的输出

        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 3. 实例化模型、优化器和损失函数
input_size = X_train.shape[1]  # 特征的维度
hidden_size = 256
output_size = 6

model = STMLModel(input_size, hidden_size, output_size)
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-6)
criterion = nn.CrossEntropyLoss()


train_losses = []
train_accuracies = []
epochs = 1000  # 训练次数

# 定义日志文件的路径
log_dir = '../logs'
log_path = os.path.join(log_dir, f'LSTM_training_logs_Epochs{epochs}.txt')

# 检查日志目录是否存在，如果不存在，则创建它
if not os.path.exists(log_dir):
    os.makedirs(log_dir)



# 将数据写入日志文件中
with open(log_path, 'w') as f:
    best_val_accuracy = 0
    # 将数据写入日志文件中
    for epoch in range(epochs):
        # 打印每一轮信息
        print(f"Epoch {epoch + 1}")
        f.write(f"Epoch {epoch + 1}\n")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        scheduler.step(epoch_loss)
        f.write(f"Training Loss: {epoch_loss}\n")

        # 更新学习率
        scheduler.step(epoch_loss)
        epoch_accuracy = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        test_accuracy = correct / total
        train_accuracies.append(test_accuracy)
        print(f'Test Accuracy: {test_accuracy:.4f}%')

        f.write(f"Test Accuracy: {test_accuracy}\n")

        # 使用 Scikit-learn 计算精确率、召回率和 F1 分数
        report = classification_report(true_labels, predicted_labels, output_dict=True)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')

        print("Macro-average metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        f.write("Macro-average metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

        true_labels_onehot = np.eye(6)[true_labels]
        predicted_scores = np.eye(6)[predicted_labels]

        # 保存最优模型
        if test_accuracy > best_val_accuracy:
            best_val_accuracy = test_accuracy
            torch.save(model.state_dict(), '../saveModel/LSTM_best_model.pth')

            data_dict = {
                'True Labels': true_labels,
                'Predicted Labels': predicted_labels
            }
            df = pd.DataFrame(data_dict)
            # 如果是第一次写入，写入列名
            df.to_csv('../labels_and_result/LSTM_labels_and_Predicted.csv')

            # 计算混淆矩阵
            cm = confusion_matrix(true_labels, predicted_labels)

            # 绘制混淆矩阵
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Move1', 'Move2', 'Move3', 'Move4', 'Move5', 'Move6'],
                        yticklabels=['Move1', 'Move2', 'Move3', 'Move4', 'Move5', 'Move6'])
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')

            # 保存混淆矩阵为PDF文件
            plt.savefig('../confusionMatrix/LSTM_confusion_matrix.pdf', format='pdf', bbox_inches='tight')

        mAP = average_precision_score(true_labels_onehot, predicted_scores, average='macro')
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")

        # 记录分隔线
        print("-----------------------------------")
        f.write("-----------------------------------\n")


def visualize_data(data, labels, class_names):
    num_samples = min(5, len(data))  # 最大可视化数量
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        for i in range(num_samples):
            print(f"Sample {i+1}:")
            print("Label:", labels[i])
            print("Class name:", class_names)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(data[i])
            plt.title(f"Waveform - Class {class_names[labels[i]]}")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")

            # 将单个样本包装成一个 batch（增加 batch 维度）
            input_tensor = torch.tensor(data[i], dtype=torch.float32).unsqueeze(0)  # 增加 batch 维度
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().detach().numpy()

            plt.subplot(1, 2, 2)
            plt.bar(range(len(class_names)), probabilities)
            plt.xticks(range(len(class_names)), class_names)
            plt.title("Predicted Probabilities")
            plt.xlabel("Class")
            plt.ylabel("Probability")

        plt.tight_layout()
        plt.show()



class_names = ['Move 1', 'Move 2', 'Move 3', 'Move 4', 'Move 5', 'Move 6']
visualize_data(X_test, y_test, class_names)
