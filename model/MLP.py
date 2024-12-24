import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 1. 数据加载和预处理
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

            # 统计特征
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            max_val = np.max(window, axis=0)
            min_val = np.min(window, axis=0)
            skewness = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 3), axis=0, arr=window)
            kurtosis = np.apply_along_axis(lambda x: np.mean(((x - np.mean(x)) / np.std(x)) ** 4), axis=0, arr=window)

            # 结合统计特征和频域特征（FFT）
            fft_data = np.fft.fft(window, axis=0)
            fft_magnitude = np.abs(fft_data)[:len(fft_data)//2]
            augmented_fft = add_noise(fft_magnitude)

            feature_vector = np.concatenate([mean, std, max_val, min_val, skewness, kurtosis, augmented_fft.flatten()])

            data.append(feature_vector)
            labels.append(move_num - 1)
    return np.array(data), np.array(labels)

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

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建DataLoader用于训练和测试
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x

# 2. 定义训练函数
input_size = X_train.shape[1]
hidden_size = 512
output_size = 6

model = MLPModel(input_size, hidden_size, output_size)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # 使用 AdamW 优化器
criterion = nn.CrossEntropyLoss()

# 定义学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)


train_losses = []
train_accuracies = []
epochs = 1000  # 训练次数

# 定义日志文件的路径
log_dir = '../logs'
log_path = os.path.join(log_dir, f'MLP_training_logs_Epochs{epochs}.txt')

# 检查日志目录是否存在，如果不存在，则创建它
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 将数据写入日志文件中
with open(log_path, 'w') as f:
    best_val_accuracy = 0
    for epoch in range(epochs):
        # 打印每一轮信息
        print(f"Epoch {epoch + 1}")
        f.write(f"Epoch {epoch + 1}\n")

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
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

        epoch_accuracy = 100 * correct / total

        model.eval()
        correct = 0
        total = 0
        predicted_labels = []
        true_labels = []
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
            torch.save(model.state_dict(), '../saveModel/MLP_best_model.pth')

            data_dict = {
                'True Labels': true_labels,
                'Predicted Labels': predicted_labels
            }
            df = pd.DataFrame(data_dict)
            # 如果是第一次写入，写入列名
            df.to_csv('../labels_and_result/MLP_labels_and_Predicted.csv')

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
            plt.savefig('../confusionMatrix/MLP_confusion_matrix.pdf', format='pdf', bbox_inches='tight')

        mAP = average_precision_score(true_labels_onehot, predicted_scores, average='macro')
        print(f"Mean Average Precision (mAP): {mAP:.4f}")
        f.write(f"Mean Average Precision (mAP): {mAP:.4f}\n")

        # Log the separator line
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
