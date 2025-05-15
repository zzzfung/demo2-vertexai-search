# 使用官方的 Python 3.11 作为基础镜像
FROM python:3.11-slim-buster

# 设置时区为 UTC+8 (例如: Asia/Shanghai)
# 这个环境变量会被很多程序读取，包括 Python 的 datetime 模块
ENV TZ=Asia/Shanghai
# 设置工作目录
WORKDIR /app

# 将 requirements.txt 复制到工作目录
COPY requirements.txt .

# 安装依赖项
RUN pip install --no-cache-dir -r requirements.txt

# 将应用代码复制到工作目录
COPY app.py .

# 设置环境变量 (可选，但推荐)
ENV FLASK_APP=app.py

# 暴露端口
EXPOSE 8080

# 定义启动命令 (使用 Gunicorn)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]