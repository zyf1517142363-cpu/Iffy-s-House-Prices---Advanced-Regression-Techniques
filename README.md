# House Prices - Advanced Regression Techniques

Kaggle 房价预测项目：训练模型、生成提交文件，并提供预测 API。

## 项目结构

- `data/`：训练与测试数据、提交文件
- `models/`：训练好的模型与元数据
- `reports/`：训练指标与运行信息
- `src/`：训练、预测与 API 代码

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型（含模型对比）

```bash
python src/train_model.py --compare
```

输出：
- 模型文件：`models/model.joblib`
- 模型元数据：`models/model_meta.json`
- 指标：`reports/metrics.json`
- 运行信息：`reports/run.json`

### 3. 生成测试集预测

```bash
python src/predict.py
```

输出：
- `data/submission.csv`

## API 服务

### 1. 启动 API

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### 2. 预测请求

支持以下输入格式：
- `{"record": {...}}`
- `{"records": [{...}, {...}]}`
- 单条 dict 或 list

示例：
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records":[{"OverallQual":7,"GrLivArea":1710,"YearBuilt":2003,"TotalBsmtSF":856}]}'
```

响应：
```json
{"predictions": [123456.78]}
```

## 说明

- 训练默认对目标值使用 `log1p` 变换，适合 Kaggle RMSLE 评估。
- API 会根据 `models/model_meta.json` 的特征列表对齐输入字段，缺失字段自动填充为缺失值。
