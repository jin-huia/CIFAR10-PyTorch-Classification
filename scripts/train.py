import sys
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

# Windows路径修复
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from datasets.cifar10_dataset import get_cifar10_loaders
from models.resnet18 import get_model
from losses.loss_fn import get_loss_fn
from tools.utils import plot_all_curves

def main():
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_precision_list = []
    val_recall_list = []
    cfg_path = os.path.join(BASE_DIR, "configs", "train_config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"====> device = {device} , cuda available: {torch.cuda.is_available()}")
    batch_size = cfg["batch_size"]
    epoch_num = cfg["epoch"]
    lr = cfg["lr"]
    data_root = cfg["data_root"]
    save_dir = cfg["save_dir"]

    # 1 数据集
    train_loader, val_loader = get_cifar10_loaders(data_root, batch_size)

    # 2 模型
    model = get_model()
    model = model.to(device)

    # 3 loss、优化器
    criterion = get_loss_fn()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=160) 


    train_acc_list = []
    val_acc_list = []
    train_loss_list = []

    for epoch in range(epoch_num):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum= 0.0

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epoch_num}")
        for imgs, labels in pbar:
            # =====【重点！把数据搬到GPU/CPU】=====
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            _, pred = torch.max(out.data, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()
            train_loss_sum += loss.item()*labels.size(0)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})


        train_acc = train_correct / train_total
        avg_train_loss = train_loss_sum / train_total
        train_acc_list.append(train_acc)
        train_loss_list.append(avg_train_loss)

        scheduler.step()  # 更新学习率

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                out = model(imgs)
                _, pred = torch.max(out, dim=1)
                val_total += labels.size(0)
                val_correct += torch.sum(pred == labels).item()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
        val_acc = val_correct / val_total
        val_acc_list.append(val_acc)

        
        p=precision_score(all_labels, all_preds, average='macro', zero_division=0)
        r=recall_score(all_labels, all_preds, average='macro')
        val_precision_list.append(p)
        val_recall_list.append(r)
        print(f"epoch:{epoch+1}  train_acc:{train_acc:.4f}  val_acc:{val_acc:.4f}")

    
    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "resnet18_cifar10.pth"))
    print("=====开始绘图=====")
    print(f"loss:{len(train_loss_list)}, train_acc:{len(train_acc_list)}, val_acc:{len(val_acc_list)}, precision:{len(val_precision_list)}, recall:{len(val_recall_list)}")
    plot_all_curves(
    train_loss_list,
    train_acc_list,
    val_acc_list,
    val_precision_list,
    val_recall_list,
    os.path.join(save_dir, "all_curves.png")
)
    print("=====绘图完成，图片已保存=====")

if __name__ == "__main__":
    main()
