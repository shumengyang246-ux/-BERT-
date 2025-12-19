***项目结构***
SoftMaskedBERT
├── data/                   # 数据存放
│   ├── sighan/             # SIGHAN 13/14/15 原始数据
│   ├── processed/          # 预处理后的数据 (pkl, json等)
├── checkpoints/            # 模型权重保存与日志
├── src/                    # 核心代码
│   ├── dataset.py          # Dataset类，处理Tokenization和数据增强
│   ├── model.py            # SoftMaskedBERT 模型架构定义 (核心)
│   ├── train.py            # 训练循环、Loss计算、验证逻辑
│   ├── pre_train.py        # 先对模型在7w多条自己构造的数据集上进行第一轮训练
|   ├── finetune_sighan.py  # 加载第一轮训练好的权重，在sighan 6k多条数据上进行第二轮训练
|   ├── inference.py        # 加载第二轮训练好的模型权重，模型输出引擎
|   ├── config.py           # 超参数配置 (lr, batch_size, lambda等)
|   ├── api.py              # 使用FastAPI库，将模型写入后端，便于前端展示
|   └── test.py             # 交互式测试模型实际效果
├── figure/                 # 图片保存
├── web/                    # 前后端脚本
├── final_model/            # 最终训练好的模型权重与日志
└── 文本纠错说明.txt         # 模型实际效果测试说明


***所需依赖***
列出虚拟环境中安装的必要库：
click                     8.3.0                    pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
cuda-cccl                 12.9.27                       0    nvidia
cuda-cccl_win-64          12.9.27                       0    nvidia
cuda-cudart               12.4.127                      0    nvidia
cuda-cudart-dev           12.4.127                      0    nvidia
cuda-cupti                12.4.127                      0    nvidia
cuda-libraries            12.4.1                        0    nvidia
cuda-libraries-dev        12.4.1                        0    nvidia
cuda-nvrtc                12.4.127                      0    nvidia
cuda-nvrtc-dev            12.4.127                      0    nvidia
cuda-nvtx                 12.4.127                      0    nvidia
cuda-opencl               12.9.19                       0    nvidia
cuda-opencl-dev           12.9.19                       0    nvidia
cuda-profiler-api         12.9.79                       0    nvidia
cuda-runtime              12.4.1                        0    nvidia
cuda-version              12.9                          3    nvidia
datasets                  3.3.2           py312haa95532_0
dill                      0.3.8           py312haa95532_0
fastapi                   0.124.4                  pypi_0    pypi
huggingface_hub           0.34.4          py312haa95532_0
numpy                     1.26.3                   pypi_0    pypi
numpy-base                1.26.4          py312h4dde369_0
openai                    1.109.1         py312haa95532_0
pandas                    2.2.2                    pypi_0    pypi
pip                       23.3.1          py312haa95532_0
python                    3.12.3               h1d929f7_0
python-dateutil           2.9.0post0      py312haa95532_2
python-fastjsonschema     2.20.0          py312haa95532_0
python-json-logger        3.2.1           py312haa95532_0
python-tzdata             2025.2             pyhd3eb1b0_0
python-xxhash             3.5.0           py312h827c3e9_0
pytorch                   2.5.0           py3.12_cuda12.4_cudnn9_0    pytorch
pytorch-cuda              12.4                 h3fd98bf_7    pytorch
pytorch-mutex             1.0                        cuda    pytorch
requests                  2.32.5                   pypi_0    pypi
tokenizers                0.22.1                   pypi_0    pypi
torchaudio                2.5.0                    pypi_0    pypi
torchvision               0.20.0                   pypi_0    pypi
tornado                   6.4.2           py312h827c3e9_0
tqdm                      4.67.1          py312hfc267ef_0
traitlets                 5.14.3          py312haa95532_0
transformers              4.56.2                   pypi_0    pypi

***快速启动***
按照路径存放好文件
快速测试模型：(虚拟环境下)python test.py
前端展示：(虚拟环境下)python.exe -m uvicorn api:app --host 0.0.0.0 --port 8000
之后运行：http://localhost:8000



