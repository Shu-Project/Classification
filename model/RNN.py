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
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 滑动平均滤波（填充零，确保数据长度一致）
def smooth_data(data, window_size=5):
    smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return smoothed


# 对数据中的每一列（即每个通道数据）进行平滑处理
def preprocess_data_with_smoothing(data, window_size=5):
    smoothed_data = []
    # 对每个特征（通道）进行处理
    for i in range(data.shape[1]):
        smoothed_data.append(smooth_data(data[:, i], window_size))
    return np.column_stack(smoothed_data)

# 加载并处理数据
def load_data():
    data = []
    labels = []
    for move_num in range(1, 7):
        move_file = f"../Datas/Move{move_num}.xlsx"
        df = pd.read_excel(move_file)

        time_series = df['Time'].values
        channels = df.drop(columns=['Time']).values

        for start in range(0, len(time_series) - 80, 40):
            window = channels[start:start + 80]

            # 计算统计特征
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            max_val = np.max(window, axis=0)
            min_val = np.min(window, axis=0)
            skewness = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 3), axis=0, arr=window)
            kurtosis = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 4), axis=0, arr=window)

            # FFT 和功率谱密度
            fft_data = np.fft.fft(window, axis=0)
            fft_magnitude = np.abs(fft_data)[:len(fft_data)//2]
            augmented_fft = add_noise(fft_magnitude)

            feature_vector = np.concatenate([mean, std, max_val, min_val, skewness, kurtosis, augmented_fft.flatten()])

            data.append(feature_vector)
            labels.append(move_num - 1)

    data = np.array(data)
    labels = np.array(labels)

    # 在这里进行平滑处理
    data = preprocess_data_with_smoothing(data, window_size=5)

    return data, labels



# 数据增强 添加噪声
def add_noise(data, noise_factor=0.01):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

# 加载数据
data, labels = load_data()

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# 分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 计算输入特征的大小，确保可以被80整除
num_features = X_train.shape[1]
if num_features % 80 != 0:
    # 使用零填充，使得特征数量可以被80整除
    padding_size = 80 - (num_features % 80)
    X_train = np.pad(X_train, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
    X_test = np.pad(X_test, ((0, 0), (0, padding_size)), mode='constant', constant_values=0)
    num_features = X_train.shape[1]

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 80, num_features // 80)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 80, num_features // 80)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader用于训练和测试
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=6, num_layers=2, dropout=0.3):
        super(RNNModel, self).__init__()
        # 使用 GRU
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)  # GRU 层的输出
        out = out[:, -1, :]  # 只取最后一个时间步的输出
        out = self.dropout(out)  # Dropout 层
        out = self.fc(out)  # 全连接层
        return out


# 实例化模型
input_size = X_train_tensor.shape[2]  # 输入特征大小
model = RNNModel(input_size=input_size, hidden_size=128, num_classes=6, num_layers=3, dropout=0.3)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# 修改优化器（AdamW，适用于权重衰减）
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 调整学习率调度器，增加步数调整
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练过程中的日志
train_losses = []
train_accuracies = []
# 训练次数
epochs = 1000

# 定义日志文件的路径
log_dir = '../logs'
log_path = os.path.join(log_dir, f'RNN_training_logs_Epochs{epochs}.txt')

# 检查日志目录是否存在，如果不存在，则创建它
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 将数据写入日志文件中
with open(log_path, 'w') as f:
    best_val_accuracy = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        f.write(f"Epoch {epoch + 1}\n")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()  # 清空梯度
            outputs = model(inputs)  # 向前传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        f.write(f"Training Loss: {epoch_loss}\n")

        epoch_accuracy = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
        with torch.no_grad():  # 禁用梯度计算
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
        report = classification_report(true_labels, predicted_labels, output_dict=True, zero_division=1)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='macro',
                                                                   zero_division=1)

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
            torch.save(model.state_dict(), '../saveModel/RNN_best_model.pth')

            data_dict = {
                'True Labels': true_labels,
                'Predicted Labels': predicted_labels
            }
            df = pd.DataFrame(data_dict)
            # 如果是第一次写入，写入列名
            df.to_csv('../labels_and_result/RNN_labels_and_Predicted.csv')

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
            plt.savefig('../confusionMatrix/RNN_confusion_matrix.pdf', format='pdf', bbox_inches='tight')


        mAP = average_precision_score(true_labels_onehot, predicted_scores, average='macro')
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")

        scheduler.step()  # 更新学习率

        # 记录分割线
        print("-----------------------------------")
        f.write("-----------------------------------\n")

# 可视化函数
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
            # 调整输入形状，使其符合 RNN 输入要求: (batch_size, sequence_length, input_size)
            input_tensor = input_tensor.reshape(1, 80, num_features // 80)  # 确保每个时间步是一个特征

            # 进行预测
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

