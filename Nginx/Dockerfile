# 使用官方的 Nginx 镜像，alpine 版本更小巧
FROM nginx:stable-alpine

# Nginx 默认的静态文件根目录是 /usr/share/nginx/html
# 将你的 HTML, CSS, JS 和图片文件复制到该目录

# 清理掉 Nginx 默认的欢迎页面（可选，但推荐，以防万一）
RUN rm /usr/share/nginx/html/index.html /usr/share/nginx/html/50x.html

# 复制你的网站文件
COPY index.html /usr/share/nginx/html/
COPY style.css /usr/share/nginx/html/
COPY script.js /usr/share/nginx/html/

COPY placeholder_*.png /usr/share/nginx/html/


# 暴露容器的 80 端口，Nginx 默认监听此端口
EXPOSE 80