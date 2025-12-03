# 🌾 基于YOLO和大语言模型的小麦病虫害识别系统

代码地址：[https://mbd.pub/o/bread/mbd-YZWZmJhyaA==](https://mbd.pub/o/bread/mbd-YZWZmJhyaA==)

**基于深度学习的小麦病虫害智能检测与诊断系统**

[功能特性](#-功能特性) • [技术架构](#-技术架构) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [开发文档](#-开发文档)

</div>

---

## 📋 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [技术架构](#-技术架构)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [使用指南](#-使用指南)
- [API 文档](#-api-文档)
- [项目结构](#-项目结构)
- [开发指南](#-开发指南)
- [常见问题](#-常见问题)

---

## 🎯 项目简介

小麦病虫害智能识别系统是一个结合了 **YOLO 目标检测**和 **LLaVA 多模态 AI** 的农业科技应用，旨在为农业生产提供精准、快速的病虫害诊断服务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4ddece36676346ae810fa428c8b99439.png#pic_center)

### 核心优势

- 🚀 **实时检测**：基于 YOLOv8 的高效目标检测，毫秒级响应
- 🤖 **AI 问答**：集成 LLaVA 多模态模型，提供专业的诊断建议
- 📊 **可视化界面**：现代化的 Web 界面，支持拖拽上传、实时标注
- 🎓 **可训练**：支持自定义数据集训练，适应不同地区病害特征
- 📱 **响应式设计**：完美适配桌面、平板和移动设备
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/05dee8f579904dad9d6febad4ddcbf8e.png#pic_center)

### 支持的病害类型

| 病害类型   | 英文名称             | 检测能力 |
| ---------- | -------------------- | -------- |
| 小麦白粉病 | Powdery Mildew       | ✅ 高精度 |
| 小麦叶锈病 | Leaf Rust            | ✅ 高精度 |
| 小麦叶斑病 | Septoria Leaf Blotch | ✅ 高精度 |
| 小麦赤霉病 | Fusarium Head Blight | ✅ 高精度 |
| 小麦黄斑病 | Tan Spot             | ✅ 高精度 |
| 蚜虫       | Aphids               | ✅ 高精度 |
| 健康小麦   | Healthy Wheat        | ✅ 高精度 |

---

## ✨ 功能特性

### 1. 智能诊断系统

- **图像上传**：支持拖拽上传，实时预览
- **目标检测**：自动识别图片中的病害区域
- **置信度评分**：显示每个检测结果的可信度
- **边界框标注**：彩色标注检测到的病害位置
- **多目标检测**：一张图片可同时检测多个病害

### 2. AI 智能问答

- **RAG 增强**：结合病害知识库提供精准回答
- **多轮对话**：支持连续提问，记录对话历史
- **图像理解**：基于 LLaVA 的视觉语言模型
- **专业建议**：提供防治方法、用药指导等专业信息
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/443c9c66e5e64199bb5b0cff7b034da9.png#pic_center)

### 3. 历史记录管理

- **记录查询**：查看所有历史检测记录
- **详细信息**：包含检测时间、结果、AI 分析
- **数据导出**：支持删除不需要的记录
- **图片缩略图**：快速浏览历史图片
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5e34a8de11dd47efba8a9cf5c0ba5696.png#pic_center)

### 4. 模型训练系统

- **在线训练**：直接在 Web 界面配置训练参数
- **实时监控**：终端式日志显示，查看训练进度
- **指标展示**：mAP、精确率、召回率等关键指标
- **训练历史**：保存所有训练结果，支持模型对比
- **数据集管理**：支持从 Roboflow 下载数据集

### 5. 精美的 UI/UX

- **玻璃拟态设计**：现代化的视觉效果
- **渐变色彩**：丰富的色彩搭配
- **流畅动画**：`fade-in`、`scale`、`shimmer` 等动画效果
- **响应式布局**：完美适配各种屏幕尺寸
- **暗色背景**：护眼的深色主题

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4137a1feb5974fa5b226ad108698cefe.png#pic_center)

## 🏗️ 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (React + Vite)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Dashboard  │  │   Training   │  │   History    │  │
│  │  智能诊断页   │  │   模型训练    │  │  历史记录     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP/REST API
┌────────────────────────┴────────────────────────────────┐
│                  后端 (FastAPI + Python)                 │
│  ┌───────────────────────────────────────────────────┐  │
│  │              API 路由层                           │  │
│  │  /detect  /analyze  /train  /history  /dataset   │  │
│  └───────────────────────────────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ YOLODetector │  │ LLaVAAnalyzer│  │TrainingMgr   │  │
│  │  YOLO检测    │  │  AI多模态    │  │  模型训练     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│  ┌──────────────────────────────────────────────────┐  │
│  │           数据库层 (SQLite + SQLAlchemy)          │  │
│  │    DetectionRecord  |  DiseaseInfo               │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  AI 模型层                              │
│  ┌──────────────┐  ┌──────────────┐                    │
│  │   YOLOv8     │  │    LLaVA     │                    │
│  │  目标检测模型  │  │ 视觉语言模型  │                    │
│  └──────────────┘  └──────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

### 技术栈详情

#### 前端技术
- **框架**: React 18 + Vite
- **样式**: Tailwind CSS + 自定义 CSS
- **状态管理**: React Hooks (useState, useEffect, useRef)
- **HTTP 客户端**: Axios
- **路由**: React Router DOM
- **UI 组件**: 自定义组件（玻璃拟态、渐变卡片等）

#### 后端技术
- **Web 框架**: FastAPI 0.100+
- **深度学习**: 
  - Ultralytics YOLOv8（目标检测）
  - Ollama + LLaVA（多模态 AI）
- **数据库**: SQLite + SQLAlchemy ORM
- **图像处理**: OpenCV, Pillow
- **并发**: Python Threading（训练任务）

#### AI 模型
- **YOLO**: Ultralytics YOLOv8（n/s/m/l/x 多种尺寸）
- **LLaVA**: 多模态视觉语言模型（通过 Ollama 部署）

---

## 💻 系统要求

### 硬件要求
- **CPU**: 4 核心及以上（推荐 8 核心）
- **内存**: 8GB RAM（推荐 16GB）
- **GPU**: NVIDIA GPU 推荐（用于训练），CPU 推理也可
- **存储**: 至少 10GB 可用空间

### 软件要求
- **操作系统**: Windows 10/11, macOS, Linux
- **Python**: 3.8 或更高版本
- **Node.js**: 16.0 或更高版本
- **npm/pnpm**: 最新版本
- **Ollama**: 用于运行 LLaVA 模型

---

## 🚀 快速开始


### 1. 安装 Ollama 和 LLaVA

```bash
# 安装 Ollama
curl https://ollama.ai/install.sh | sh  # Linux/macOS
# 或访问 https://ollama.ai/download 下载 Windows 版本

# 拉取 LLaVA 模型
ollama pull llava
```

### 2. 后端设置

```bash
cd backend

# 安装依赖
pip install -r requirements.txt

# 启动后端服务
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

后端服务将运行在 `http://localhost:8000`

### 3. 前端设置

```bash
cd frontend

# 安装依赖
npm install
# 或使用 pnpm
pnpm install

# 启动开发服务器
npm run dev
```

前端服务将运行在 `http://localhost:5173`

### 5. 访问应用

在浏览器打开 `http://localhost:5173`，开始使用系统！

---

## 📖 使用指南

### 智能诊断

1. **上传图片**
   - 点击上传区域或拖拽图片到上传框
   - 支持 JPG、PNG 格式
   - 推荐分辨率：640x640

2. **开始检测**
   - 点击"开始 AI 诊断"按钮
   - 等待几秒钟，系统会自动检测病害
   - 检测结果显示在图片上方

3. **AI 问答**
   - 在右侧对话框输入问题
   - 例如："如何防治这种病害？"
   - AI 会基于检测结果和知识库回答

### 模型训练

1. **准备数据集**
   - 使用 YOLO 格式的数据集
   - 包含 `data.yaml` 配置文件
   - 或从 Roboflow 下载

2. **配置参数**
   - 设置训练轮数（epochs）
   - 选择批次大小（batch size）
   - 选择模型尺寸（n/s/m/l/x）

3. **开始训练**
   - 点击"开始训练"按钮
   - 在终端查看实时日志
   - 训练完成后查看指标

### 历史记录

- 查看所有诊断记录
- 点击记录查看详情
- 删除不需要的记录
- 导出数据（计划中）

---



## 📁 项目结构

```
XM/
├── backend/                    # 后端代码
│   ├── main.py                # FastAPI 主应用
│   ├── engine.py              # YOLO 和 LLaVA 引擎
│   ├── training.py            # 训练管理器
│   ├── models.py              # 数据库模型
│   ├── database.py            # 数据库配置
│   ├── dataset_service.py     # 数据集管理
│   ├── requirements.txt       # Python 依赖
│   ├── model.pt               # YOLO 模型文件
│   ├── uploads/               # 上传的图片
│   ├── runs/                  # 训练结果
│   └── datasets/              # 数据集目录
│
├── frontend/                   # 前端代码
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx  # 主诊断页面
│   │   │   ├── Training.jsx   # 训练页面
│   │   │   └── History.jsx    # 历史记录页面
│   │   ├── App.jsx            # 应用主组件
│   │   └── index.css          # 全局样式
│   ├── public/                # 静态资源
│   ├── package.json           # Node 依赖
│   ├── vite.config.js         # Vite 配置
│   ├── tailwind.config.js     # Tailwind 配置
│   └── postcss.config.js      # PostCSS 配置
│
├── data/
│   └── diseases.json          # 病害知识库
│
└── README.md                   # 本文件
```

---

## 🛠️ 开发指南

### 添加新的病害类型

1. **更新数据库**
   ```python
   # backend/main.py - DISEASE_NAME_MAP
   DISEASE_NAME_MAP = {
       "new_disease": "新病害中文名",
       # ...
   }
   ```

2. **更新知识库**
   ```json
   // data/diseases.json
   {
     "name": "新病害",
     "symptoms": "症状描述",
     "treatment": "防治方法"
   }
   ```

3. **重新训练模型**
   - 准备包含新病害的数据集
   - 使用训练页面重新训练

### 自定义 UI 主题

编辑 `frontend/src/index.css`：

```css
:root {
  --primary-color: #your-color;
  --gradient-start: #your-gradient-start;
  --gradient-end: #your-gradient-end;
}
```

### 部署到生产环境

```bash
# 后端
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# 前端
cd frontend
npm run build
# 将 dist/ 目录部署到 Nginx 或其他 Web 服务器
```

---

## ❓ 常见问题

### Q: 为什么检测不出病害？
A: 
- 确保模型文件 `model.pt` 存在
- 检查图片质量和分辨率
- 确认病害类型在支持列表中

### Q: AI 不回答问题？
A: 
- 确认 Ollama 服务正在运行（`ollama serve`）
- 确认已安装 LLaVA 模型（`ollama list`）
- 检查后端日志是否有错误

### Q: 训练速度很慢？
A: 
- 使用 GPU 可大幅提升速度
- 减小 batch_size
- 选择较小的模型（yolov8n）

### Q: 如何更新模型？
A: 
- 准备新数据集
- 使用训练页面重新训练
- 训练完成后，`best.pt` 会自动复制为 `model.pt`

---
