# demo2_vertexai_search

# 概述

本项目的核心目标，是借助Vertex AI Vector Search这一先进技术，精心打造一个功能强大的搜索引擎。Vertex AI Vector Search技术以其独特的优势，能够深度理解语义和精准捕捉相似性。通过运用该技术，所构建的搜索引擎能够依据客户输入的内容，从语义和相似性的维度展开全面检索。相较于传统的搜索方式，它不再仅仅局限于简单的关键词匹配，而是深入挖掘客户输入背后的真实意图，从而实现更为精准的商品搜索，为客户提供更优质、更符合需求的搜索结果，助力客户在海量商品信息中快速找到心仪的商品。 

# 入门

## 先决条件

开始之前，请确保已安装以下软件：

- Python 3.11 或更高版本
- pip（Python 包安装程序）

权限认证：

- Service Account（授予Vertex AI Administrator角色）

## 安装

### 1.克隆仓库

```bash
git clone https://github.com/zzzfung/demo2-vertexai-search.git

cd ./demo2-vertexai-search/Application/
```

### 2.创建虚拟环境：

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3.安装所需的软件包：

```
pip install -r requirements.txt
```

###  4.设置服务账号

```bash
# 示例：Linux 或 macOS
export GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
```

将 `KEY_PATH` 替换为包含凭据的 JSON 文件的路径。

```bash
# 示例：Windows，（PowerShell）
$env:GOOGLE_APPLICATION_CREDENTIALS="KEY_PATH"
```

将 `KEY_PATH` 替换为包含凭据的 JSON 文件的路径。

### 5.✅运行项目

```bash
python manage.py runserver
```

项目将运行在 `http://127.0.0.1:8000/`。

## 接口说明

#### 1. **预测接口**

**URL**: `http://127.0.0.1:8000/v1/predict/`

**方法**: `POST`

**描述**: 该接口接收一组产品和用户信息，返回预测结果。

**请求参数**（JSON格式）：

- `Product_ID`: 产品ID（字符串，最大长度100）
- `Gender`: 性别（字符串，最大长度1）
- `Age`: 年龄（字符串，最大长度10）
- `Occupation`: 职业（字符串，最大长度10）
- `City_Category`: 城市类别（字符串，最大长度1）
- `Stay_In_Current_City_Years`: 在当前城市居住的年数（字符串，最大长度10）
- `Marital_Status`: 婚姻状况（字符串，最大长度1）
- `Product_Category_1`: 产品类别1（字符串，最大长度10）
- `Product_Category_2`: 产品类别2（字符串，最大长度10）
- `Product_Category_3`: 产品类别3（字符串，最大长度10）

```json
# 示例请求
{
    "Product_ID": "P00069042",
    "Gender": "F",
    "Age": "26-35",
    "Occupation": "1",
    "City_Category": "A",
    "Stay_In_Current_City_Years": "2",
    "Marital_Status": "0",
    "Product_Category_1": "3",
    "Product_Category_2": "1",
    "Product_Category_3": "3"
}
```

## License

基于Apache 2.0协议， 详细请参考[LICENSE](https://github.com/zzzfung/demo2-vertexai-search/blob/main/LICENSE)

