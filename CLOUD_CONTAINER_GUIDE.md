# 云平台容器训练完整指南（新手版）

## 📋 目录
1. [准备工作](#准备工作)
2. [第一步：在本地构建Docker镜像](#第一步在本地构建docker镜像)
3. [第二步：将镜像推送到云平台](#第二步将镜像推送到云平台)
4. [第三步：在云平台创建容器](#第三步在云平台创建容器)
5. [第四步：上传数据到云平台](#第四步上传数据到云平台)
6. [第五步：在容器中运行训练](#第五步在容器中运行训练)
7. [第六步：查看训练日志](#第六步查看训练日志)
8. [常见问题解决](#常见问题解决)

---

## 准备工作

### 需要准备的东西：
1. ✅ 本地电脑（Windows/Mac/Linux都可以）
2. ✅ 安装了Docker（如果没有，看下面的安装步骤）
3. ✅ 云平台账号（阿里云/腾讯云/华为云等）
4. ✅ 训练数据（NYU数据集或其他数据）

### 安装Docker（如果还没有）

#### Windows/Mac:
1. 下载Docker Desktop: https://www.docker.com/products/docker-desktop
2. 安装并启动Docker Desktop
3. 打开命令行（Windows: PowerShell 或 CMD，Mac: Terminal）

#### Linux (Ubuntu/Debian):
```bash
# 更新软件包列表
sudo apt-get update

# 安装Docker
sudo apt-get install -y docker.io

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 验证安装（应该能看到Docker版本信息）
docker --version
```

---

## 第一步：在本地构建Docker镜像

### 1.1 打开命令行/终端

- **Windows**: 按 `Win + R`，输入 `cmd` 或 `powershell`，回车
- **Mac**: 按 `Cmd + Space`，输入 `terminal`，回车
- **Linux**: 按 `Ctrl + Alt + T`

### 1.2 进入项目目录

```bash
# 假设你的项目在 /home/jingyang/MonoLDP
cd /home/jingyang/MonoLDP

# Windows用户可能是这样：
# cd C:\Users\YourName\MonoLDP

# 确认你在正确的目录（应该能看到Dockerfile）
ls -la Dockerfile
# Windows用户用：
# dir Dockerfile
```

### 1.3 构建Docker镜像

```bash
# 构建镜像（这可能需要10-30分钟，取决于网络速度）
docker build -t monoldp:latest .

# 解释：
# - docker build: 构建镜像的命令
# - -t monoldp:latest: 给镜像起个名字叫 monoldp，标签是 latest
# - . : 表示使用当前目录的Dockerfile
```

**等待构建完成**，你会看到类似这样的输出：
```
Step 1/20 : FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
...
Successfully built abc123def456
Successfully tagged monoldp:latest
```

### 1.4 验证镜像是否构建成功

```bash
# 查看所有镜像，应该能看到 monoldp
docker images | grep monoldp

# 或者
docker images
```

你应该能看到类似这样的输出：
```
REPOSITORY   TAG      IMAGE ID       CREATED         SIZE
monoldp      latest   abc123def456   2 minutes ago   8.5GB
```

---

## 第二步：将镜像推送到云平台

### 方法A：使用阿里云容器镜像服务（推荐）

#### 2.1 登录阿里云并创建镜像仓库

1. 打开浏览器，访问：https://cr.console.aliyun.com
2. 登录你的阿里云账号
3. 点击左侧菜单 "镜像仓库" → "创建镜像仓库"
4. 填写信息：
   - **命名空间**：选择或创建一个（例如：`your-username`）
   - **仓库名称**：`monoldp`（可以自己起名）
   - **仓库类型**：选择 "私有"
   - **摘要**：可以填写 "MonoLDP训练镜像"
5. 点击 "创建镜像仓库"

#### 2.2 获取登录命令

1. 在镜像仓库页面，点击你刚创建的仓库
2. 点击 "管理" 标签
3. 找到 "登录阿里云Docker Registry" 部分
4. 复制登录命令，类似这样：
   ```bash
   sudo docker login --username=your_username registry.cn-hangzhou.aliyuncs.com
   ```
5. 在本地命令行执行这个命令，输入密码

#### 2.3 给镜像打标签

```bash
# 格式：registry.cn-区域.aliyuncs.com/命名空间/仓库名:标签
# 例如（根据你的实际情况修改）：
docker tag monoldp:latest registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
```

#### 2.4 推送镜像到阿里云

```bash
# 推送镜像（这可能需要一些时间，取决于镜像大小和网络速度）
docker push registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
```

**等待推送完成**，你会看到类似这样的输出：
```
The push refers to repository [registry.cn-hangzhou.aliyuncs.com/...]
...
latest: digest: sha256:abc123... size: 12345
```

### 方法B：使用腾讯云容器镜像服务

#### 2.1 登录腾讯云并创建镜像仓库

1. 访问：https://console.cloud.tencent.com/tcr
2. 登录腾讯云账号
3. 创建命名空间和镜像仓库（步骤类似阿里云）

#### 2.2 登录并推送

```bash
# 登录（根据你的区域修改）
docker login ccr.ccs.tencentyun.com

# 打标签
docker tag monoldp:latest ccr.ccs.tencentyun.com/your-namespace/monoldp:latest

# 推送
docker push ccr.ccs.tencentyun.com/your-namespace/monoldp:latest
```

### 方法C：使用Docker Hub（国际用户）

```bash
# 登录Docker Hub
docker login

# 打标签（your-username替换为你的Docker Hub用户名）
docker tag monoldp:latest your-username/monoldp:latest

# 推送
docker push your-username/monoldp:latest
```

---

## 第三步：在云平台创建容器

### 方法A：使用阿里云容器服务ACK

#### 3.1 创建Kubernetes集群（如果还没有）

1. 访问：https://cs.console.aliyun.com
2. 点击 "创建Kubernetes托管版集群"
3. 填写基本信息：
   - **集群名称**：`monoldp-cluster`
   - **地域**：选择离你近的区域
   - **Kubernetes版本**：选择最新稳定版
   - **节点配置**：选择GPU节点（例如：ecs.gn6i-c4g1.xlarge）
4. 点击 "创建集群"，等待创建完成（约5-10分钟）

#### 3.2 创建Pod（容器）

1. 在集群页面，点击 "工作负载" → "无状态"
2. 点击 "使用镜像创建"
3. 填写信息：
   - **应用名称**：`monoldp-training`
   - **镜像名称**：`registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest`
   - **镜像拉取策略**：选择 "Always"
   - **资源限制**：
     - CPU: 4核
     - 内存: 16Gi
     - GPU: 1（如果有GPU节点）
   - **数据卷**：
     - 添加数据卷：选择 "使用已有存储卷" 或 "新建存储卷"
     - 挂载路径：`/workspace/data`（只读）
     - 挂载路径：`/workspace/logs`（读写）
     - 挂载路径：`/workspace/models`（读写）
4. 点击 "创建"

### 方法B：使用阿里云ECS + Docker（更简单）

#### 3.1 购买ECS实例

1. 访问：https://ecs.console.aliyun.com
2. 点击 "创建实例"
3. 选择配置：
   - **实例规格**：选择GPU实例（例如：ecs.gn6i-c4g1.xlarge）
   - **镜像**：选择 "Ubuntu 20.04" 或 "CentOS 7"
   - **存储**：至少100GB
   - **网络**：选择VPC和交换机
   - **安全组**：开放SSH端口（22）和TensorBoard端口（6006）
4. 点击 "创建实例"

#### 3.2 登录ECS并安装Docker

```bash
# 1. 使用SSH登录到ECS（Windows用户可以用PuTTY或WSL）
ssh root@your-ecs-ip

# 2. 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. 安装nvidia-docker2（用于GPU支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 4. 验证GPU支持
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

#### 3.3 从镜像仓库拉取镜像

```bash
# 登录镜像仓库（阿里云）
sudo docker login --username=your_username registry.cn-hangzhou.aliyuncs.com

# 拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

# 验证镜像
docker images | grep monoldp
```

---

## 第四步：上传数据到云平台

### 方法A：使用OSS对象存储（推荐，适合大文件）

#### 4.1 创建OSS Bucket

1. 访问：https://oss.console.aliyun.com
2. 点击 "创建Bucket"
3. 填写信息：
   - **Bucket名称**：`monoldp-data`（全局唯一，需要自己起名）
   - **地域**：选择与ECS相同的区域
   - **读写权限**：选择 "私有"
4. 点击 "确定"

#### 4.2 上传数据到OSS

**在本地电脑上**：

```bash
# 安装OSS命令行工具（如果还没有）
# 下载：https://help.aliyun.com/document_detail/120075.html

# 配置OSS（只需要配置一次）
ossutil config

# 上传数据
ossutil cp -r /path/to/your/nyu_data oss://monoldp-data/nyu_data/
```

#### 4.3 在容器中挂载OSS

在创建容器时，使用OSS挂载插件，或者：

```bash
# 在ECS上安装ossfs（OSS文件系统）
wget http://gosspublic.alicdn.com/ossfs/ossfs_1.80.6_ubuntu20.04_amd64.deb
sudo dpkg -i ossfs_1.80.6_ubuntu20.04_amd64.deb

# 配置OSS访问密钥
echo your-bucket-name:your-access-key-id:your-access-key-secret > /etc/passwd-ossfs
chmod 640 /etc/passwd-ossfs

# 挂载OSS到本地目录
mkdir -p /mnt/oss-data
ossfs your-bucket-name /mnt/oss-data -o url=oss-cn-hangzhou.aliyuncs.com
```

### 方法B：使用SCP直接上传到ECS

**在本地电脑上**：

```bash
# 压缩数据（可选，加快传输速度）
tar -czf nyu_data.tar.gz /path/to/nyu_data

# 上传到ECS
scp nyu_data.tar.gz root@your-ecs-ip:/root/

# 或者直接上传整个目录
scp -r /path/to/nyu_data root@your-ecs-ip:/root/data/
```

**在ECS上**：

```bash
# 如果上传的是压缩包，解压
cd /root
tar -xzf nyu_data.tar.gz

# 创建数据目录
mkdir -p /data/nyu_data
mv nyu_data /data/
```

### 方法C：使用云盘（适合小数据集）

1. 在ECS控制台创建云盘
2. 挂载到ECS实例
3. 格式化并挂载：
   ```bash
   # 查看云盘
   lsblk
   
   # 格式化（注意：这会清空数据！）
   sudo mkfs.ext4 /dev/vdb
   
   # 挂载
   sudo mkdir -p /data
   sudo mount /dev/vdb /data
   
   # 设置开机自动挂载
   echo '/dev/vdb /data ext4 defaults 0 0' | sudo tee -a /etc/fstab
   ```

---

## 第五步：在容器中运行训练

### 5.1 启动容器

**在ECS上执行**：

```bash
# 创建必要的目录
mkdir -p /data/logs /data/models

# 启动容器（基础版本）
docker run --gpus all -it --rm \
  --name monoldp-training \
  -v /data/nyu_data:/workspace/data/nyu_data:ro \
  -v /data/logs:/workspace/logs \
  -v /data/models:/workspace/models \
  -e CUDA_VISIBLE_DEVICES=0 \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

# 解释：
# --gpus all: 使用所有GPU
# -it: 交互式终端
# --rm: 容器退出后自动删除
# --name: 容器名称
# -v: 挂载目录（格式：主机路径:容器路径:权限）
#   :ro 表示只读
# -e: 环境变量
```

### 5.2 进入容器后验证环境

```bash
# 你应该已经在容器内了，如果没有，执行：
docker exec -it monoldp-training bash

# 验证GPU
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查数据
ls -la /workspace/data/nyu_data

# 检查工作目录
pwd  # 应该显示 /workspace/DepthEstimation
ls -la  # 应该能看到 train.py 等文件
```

### 5.3 运行训练

```bash
# 在容器内执行
cd /workspace/DepthEstimation

# 基础训练命令
python train.py \
  --data_path /workspace/data/nyu_data \
  --log_dir /workspace/logs \
  --model_name monoldp_training_$(date +%Y%m%d_%H%M%S) \
  --batch_size 4 \
  --num_epochs 15 \
  --gpu_id 0

# 完整训练命令（带所有推荐参数）
python train.py \
  --data_path /workspace/data/nyu_data \
  --log_dir /workspace/logs \
  --model_name monoldp_training_$(date +%Y%m%d_%H%M%S) \
  --batch_size 4 \
  --num_epochs 15 \
  --learning_rate 5e-5 \
  --scales 0 \
  --gpu_id 0 \
  --diffusion_steps 8 6 5 \
  --diffusion_l1_weight 2.0 \
  --diffusion_ddim_weight 1.0 \
  --smoothness_weight 0.05 \
  --plane_weight 0.5 \
  --line_weight 0.1
```

### 5.4 后台运行训练（推荐）

**方法1：使用nohup**

```bash
# 在容器内
nohup python train.py \
  --data_path /workspace/data/nyu_data \
  --log_dir /workspace/logs \
  --model_name monoldp_training \
  --batch_size 4 \
  --gpu_id 0 \
  > /workspace/logs/training.log 2>&1 &

# 查看日志
tail -f /workspace/logs/training.log
```

**方法2：使用screen（推荐）**

```bash
# 在容器内安装screen（如果没有）
apt-get update && apt-get install -y screen

# 创建screen会话
screen -S training

# 在screen中运行训练
python train.py --data_path /workspace/data/nyu_data ...

# 分离screen：按 Ctrl+A，然后按 D

# 重新连接screen
screen -r training
```

**方法3：直接后台运行容器**

```bash
# 在ECS上，不进入容器，直接运行
docker run --gpus all -d \
  --name monoldp-training \
  -v /data/nyu_data:/workspace/data/nyu_data:ro \
  -v /data/logs:/workspace/logs \
  -v /data/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest \
  bash -c "cd /workspace/DepthEstimation && \
    python train.py \
      --data_path /workspace/data/nyu_data \
      --log_dir /workspace/logs \
      --model_name monoldp_training \
      --batch_size 4 \
      --gpu_id 0"
```

---

## 第六步：查看训练日志

### 6.1 查看容器日志

```bash
# 在ECS上
docker logs -f monoldp-training

# 查看最近100行
docker logs --tail 100 monoldp-training
```

### 6.2 查看训练输出文件

```bash
# 在ECS上查看日志目录
ls -lh /data/logs/

# 查看最新的训练日志
ls -lt /data/logs/ | head -5
```

### 6.3 使用TensorBoard查看训练曲线

**方法1：在容器内启动TensorBoard**

```bash
# 在容器内
tensorboard --logdir /workspace/logs --port 6006 --host 0.0.0.0
```

**方法2：在ECS上启动TensorBoard（推荐）**

```bash
# 在ECS上，使用另一个容器运行TensorBoard
docker run -d \
  --name tensorboard \
  -p 6006:6006 \
  -v /data/logs:/workspace/logs \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest \
  tensorboard --logdir /workspace/logs --port 6006 --host 0.0.0.0
```

**方法3：在本地查看（使用SSH端口转发）**

```bash
# 在本地电脑上
ssh -L 6006:localhost:6006 root@your-ecs-ip

# 然后在浏览器访问
# http://localhost:6006
```

### 6.4 下载训练结果

```bash
# 使用SCP从ECS下载模型
scp -r root@your-ecs-ip:/data/models/* /local/path/to/save/

# 或使用OSS下载
ossutil cp -r oss://monoldp-data/models/ /local/path/to/save/
```

---

## 常见问题解决

### Q1: 构建镜像时网络很慢怎么办？

**A**: 使用国内镜像源

```dockerfile
# 在Dockerfile中添加（在RUN apt-get update之前）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list
```

### Q2: 容器启动后找不到GPU

**A**: 检查nvidia-docker2是否安装

```bash
# 在ECS上
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# 如果失败，重新安装nvidia-docker2
```

### Q3: 训练时内存不足（OOM）

**A**: 减小batch size

```bash
# 将batch_size从4改为2
--batch_size 2
```

### Q4: 数据上传很慢

**A**: 使用OSS或压缩数据

```bash
# 压缩数据
tar -czf nyu_data.tar.gz nyu_data/

# 上传压缩包
scp nyu_data.tar.gz root@your-ecs-ip:/root/

# 在ECS上解压
tar -xzf nyu_data.tar.gz
```

### Q5: 容器退出后数据丢失

**A**: 确保使用volume挂载

```bash
# 检查挂载
docker inspect monoldp-training | grep Mounts

# 确保所有重要数据都通过-v参数挂载
```

### Q6: 如何停止训练

```bash
# 停止容器
docker stop monoldp-training

# 或进入容器后按 Ctrl+C
```

### Q7: 如何查看GPU使用情况

```bash
# 在ECS上
watch -n 1 nvidia-smi

# 或在容器内
nvidia-smi -l 1
```

---

## 完整示例：从零开始

假设你是一个完全的新手，按照以下步骤：

### 步骤1：本地准备（5分钟）

```bash
# 1. 打开命令行
# 2. 进入项目目录
cd /home/jingyang/MonoLDP

# 3. 构建镜像（等待10-30分钟）
docker build -t monoldp:latest .
```

### 步骤2：推送到阿里云（10分钟）

```bash
# 1. 在浏览器创建镜像仓库（见第二步）
# 2. 登录
docker login --username=your_username registry.cn-hangzhou.aliyuncs.com

# 3. 打标签
docker tag monoldp:latest registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

# 4. 推送（等待10-30分钟）
docker push registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
```

### 步骤3：创建ECS（10分钟）

```bash
# 1. 在阿里云控制台购买ECS GPU实例
# 2. 使用SSH登录
ssh root@your-ecs-ip

# 3. 安装Docker和nvidia-docker2（见第三步）
```

### 步骤4：上传数据（根据数据大小，可能1-几小时）

```bash
# 在本地
scp -r /path/to/nyu_data root@your-ecs-ip:/data/
```

### 步骤5：运行训练（开始训练）

```bash
# 在ECS上
docker pull registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

docker run --gpus all -d \
  --name monoldp-training \
  -v /data/nyu_data:/workspace/data/nyu_data:ro \
  -v /data/logs:/workspace/logs \
  -v /data/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest \
  bash -c "cd /workspace/DepthEstimation && python train.py --data_path /workspace/data/nyu_data --log_dir /workspace/logs --model_name monoldp_training --batch_size 4 --gpu_id 0"

# 查看日志
docker logs -f monoldp-training
```

---

## 快速参考命令

### 本地操作
```bash
# 构建镜像
docker build -t monoldp:latest .

# 推送到阿里云
docker tag monoldp:latest registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
docker push registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
```

### ECS操作
```bash
# 拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

# 运行训练
docker run --gpus all -d --name monoldp-training \
  -v /data/nyu_data:/workspace/data/nyu_data:ro \
  -v /data/logs:/workspace/logs \
  -v /data/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest \
  bash -c "cd /workspace/DepthEstimation && python train.py --data_path /workspace/data/nyu_data --log_dir /workspace/logs --model_name monoldp_training --batch_size 4 --gpu_id 0"

# 查看日志
docker logs -f monoldp-training

# 停止训练
docker stop monoldp-training
```

---

## 需要帮助？

如果遇到问题：
1. 检查Docker是否正常运行：`docker --version`
2. 检查GPU是否可用：`nvidia-smi`
3. 查看容器日志：`docker logs monoldp-training`
4. 检查数据路径：`ls -la /workspace/data/nyu_data`

祝你训练顺利！🎉

