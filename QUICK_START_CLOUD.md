# 云平台快速开始指南（5分钟上手）

## 🚀 最简流程

### 第一步：本地构建并推送镜像

```bash
# 1. 进入项目目录
cd /home/jingyang/MonoLDP

# 2. 构建镜像（等待10-30分钟）
docker build -t monoldp:latest .

# 3. 登录阿里云镜像仓库（在浏览器创建仓库后）
docker login --username=your_username registry.cn-hangzhou.aliyuncs.com

# 4. 打标签并推送
docker tag monoldp:latest registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
docker push registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest
```

### 第二步：在云服务器上运行

```bash
# 1. SSH登录到你的云服务器
ssh root@your-server-ip

# 2. 安装Docker（如果还没有）
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 3. 安装nvidia-docker2（GPU支持）
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 4. 上传数据（在本地执行）
scp -r /path/to/nyu_data root@your-server-ip:/data/

# 5. 在服务器上拉取镜像
docker pull registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest

# 6. 运行训练
docker run --gpus all -d \
  --name monoldp-training \
  -v /data/nyu_data:/workspace/data/nyu_data:ro \
  -v /data/logs:/workspace/logs \
  -v /data/models:/workspace/models \
  registry.cn-hangzhou.aliyuncs.com/your-namespace/monoldp:latest \
  bash -c "cd /workspace/DepthEstimation && python train.py --data_path /workspace/data/nyu_data --log_dir /workspace/logs --model_name monoldp_training --batch_size 4 --gpu_id 0"

# 7. 查看训练日志
docker logs -f monoldp-training
```

## 📝 详细步骤请查看

完整详细指南请查看：`CLOUD_CONTAINER_GUIDE.md`

