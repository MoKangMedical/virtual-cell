# virtual-cell

🔬 单细胞基础模型Benchmark平台 — **15个模型**×26个数据集×6大任务

## 项目简介

单细胞基础模型Benchmark平台，追踪最新AI虚拟细胞研究，涵盖 Transformer/GPT/BERT/Diffusion 全架构。

### 🆕 最新收录
- **Squidiff** — Nature Methods 2026年1月封面文章，扩散模型预测细胞发育和扰动响应 ([论文](https://www.nature.com/articles/s41592-025-02877-y) | [代码](https://github.com/siyuh/Squidiff))

## 功能特性

### 核心功能
- 🏥 医疗AI核心功能
- 🔬 智能诊断与分析
- 📊 数据可视化与报告
- 🤖 多模态交互支持
- 🔒 数据安全与隐私保护

### 技术特性
- 🚀 高性能计算
- 📈 可扩展架构
- 🔄 实时数据处理
- 🌐 分布式部署
- 📱 多平台支持

## 技术栈

### 后端技术
- **框架**: Python FastAPI, Django, Flask
- **AI框架**: TensorFlow, PyTorch, Scikit-learn
- **数据库**: PostgreSQL, MongoDB, Redis
- **消息队列**: RabbitMQ, Kafka
- **容器化**: Docker, Kubernetes

### 前端技术
- **框架**: React, Vue.js, Angular
- **UI库**: Ant Design, Material-UI, Element UI
- **可视化**: D3.js, ECharts, Plotly
- **移动端**: React Native, Flutter

### 数据处理
- **分析**: Pandas, NumPy, SciPy
- **可视化**: Matplotlib, Seaborn, Plotly
- **大数据**: Spark, Hadoop
- **流处理**: Flink, Storm

## 快速开始

### 环境要求

- Python 3.9+
- Node.js 16+
- Docker 20+
- Git 2.30+

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/MoKangMedical/virtual-cell.git
cd virtual-cell
```

2. **后端设置**
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，配置数据库连接等
```

3. **前端设置**
```bash
cd frontend
npm install
npm run build
```

4. **数据库设置**
```bash
# 初始化数据库
python manage.py migrate
python manage.py createsuperuser
```

5. **启动服务**
```bash
# 使用Docker Compose（推荐）
docker-compose up -d

# 或手动启动
python manage.py runserver
```

## 项目结构

```
virtual-cell/
├── backend/                 # 后端代码
│   ├── api/                # API接口
│   ├── models/             # 数据模型
│   ├── services/           # 业务逻辑
│   ├── utils/              # 工具函数
│   └── tests/              # 测试用例
├── frontend/               # 前端代码
│   ├── src/               # 源代码
│   ├── public/            # 静态资源
│   └── package.json       # 依赖配置
├── ai-engine/             # AI引擎
│   ├── models/           # AI模型
│   ├── training/         # 训练脚本
│   └── inference/        # 推理服务
├── data/                  # 数据存储
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后的数据
│   └── models/           # 训练好的模型
├── docs/                  # 项目文档
│   ├── api/              # API文档
│   ├── user/             # 用户手册
│   └── dev/              # 开发文档
├── scripts/               # 脚本工具
│   ├── deploy/           # 部署脚本
│   ├── data/             # 数据处理脚本
│   └── utils/            # 工具脚本
├── tests/                 # 测试代码
├── docker-compose.yml     # Docker编排
├── Dockerfile            # Docker配置
├── requirements.txt      # Python依赖
├── .env.example          # 环境变量示例
├── .gitignore           # Git忽略文件
└── README.md            # 项目说明
```

## API文档

### 主要接口

#### 基础接口
- `GET /` - 首页
- `GET /health` - 健康检查
- `GET /api/v1/status` - 系统状态

#### 数据接口
- `GET /api/v1/data` - 获取数据列表
- `POST /api/v1/data` - 上传数据
- `GET /api/v1/data/<built-in function id>` - 获取特定数据

#### 分析接口
- `POST /api/v1/analyze` - 数据分析
- `GET /api/v1/analyze/<built-in function id>` - 获取分析结果
- `GET /api/v1/reports` - 获取报告列表

#### 用户接口
- `POST /api/v1/auth/login` - 用户登录
- `POST /api/v1/auth/register` - 用户注册
- `GET /api/v1/users/me` - 获取当前用户信息

### 详细文档

启动服务后，访问以下地址查看完整API文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 配置说明

### 环境变量

创建 `.env` 文件并配置以下变量：

```bash
# 基础配置
DEBUG=True
SECRET_KEY=your-secret-key
ALLOWED_HOSTS=localhost,127.0.0.1

# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379/0

# AI服务配置
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_TOKEN=your-hf-token

# 文件存储配置
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
AWS_STORAGE_BUCKET_NAME=your-bucket-name

# 邮件配置
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-email-password
```

## 部署指南

### Docker部署（推荐）

1. **构建镜像**
```bash
docker build -t virtual-cell .
```

2. **运行容器**
```bash
docker run -d -p 8000:8000 --name virtual-cell virtual-cell
```

3. **使用Docker Compose**
```bash
docker-compose up -d
```

## 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_api.py

# 生成测试覆盖率报告
python -m pytest --cov=app tests/
```

## 贡献指南

我们欢迎任何形式的贡献！请遵循以下步骤：

1. **Fork本仓库**
2. **创建特性分支**
```bash
git checkout -b feature/AmazingFeature
```

3. **提交更改**
```bash
git commit -m 'Add some AmazingFeature'
```

4. **推送到分支**
```bash
git push origin feature/AmazingFeature
```

5. **创建Pull Request**

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 联系方式

- **项目维护者**: MoKangMedical
- **邮箱**: contact@mokangmedical.com
- **项目主页**: https://github.com/MoKangMedical/virtual-cell
- **问题反馈**: https://github.com/MoKangMedical/virtual-cell/issues

## 致谢

感谢所有为这个项目做出贡献的开发者和医疗领域专家！

---

**注意**: 这是一个活跃开发中的项目，API和功能可能会发生变化。请定期查看更新日志获取最新信息。
