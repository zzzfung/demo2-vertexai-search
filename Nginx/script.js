document.addEventListener('DOMContentLoaded', () => {
    // 获取 DOM 元素
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('search-input');
    const resultsInfo = document.getElementById('results-info');
    const resultsGrid = document.getElementById('results-grid');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');

    // API 的 URL 地址
    const API_URL = 'https://flask-image-to-text-app-746866758104.us-central1.run.app/search';

    // 函数：将 gs:// URL 转换为 https:// URL
    function convertGcsToHttps(gcsUrl) {
        if (gcsUrl && gcsUrl.startsWith('gs://')) {
            // 基本转换，可能需要根据具体的存储桶/路径结构进行调整
            // 将 'gs://' 替换为 Google Cloud Storage 的公共 HTTPS 前缀
            return gcsUrl.replace('gs://', 'https://storage.googleapis.com/');
        }
        // 如果不是 gs:// URL，返回原始值或根据需要处理潜在错误
        return gcsUrl;
    }

    // 函数：显示搜索结果
    function displayResults(results) {
        // 清除之前的结果和错误信息
        resultsGrid.innerHTML = '';
        errorMessage.textContent = '';

        // 更新结果计数信息
        // 注意: API 示例不提供 'total' 总项目数，只返回列表长度。
        // 我们将模仿图片中显示的格式，但 'total' 不会准确。
        resultsInfo.textContent = `检索到 ${results.length} 个结果。`;
        // 如果知道总数： `检索到 ${results.length} 个结果，总计 X 个项目`;

        if (results.length === 0) {
            resultsGrid.innerHTML = '<p>未找到结果。</p>';
            return;
        }

        // 遍历结果并创建 DOM 元素
        results.forEach((result, index) => {
            const imageUrl = convertGcsToHttps(result.gcs_url);

            // 创建结果项的元素
            const itemDiv = document.createElement('div');
            itemDiv.classList.add('result-item');

            const img = document.createElement('img');
            img.src = imageUrl;
            // 如果 API 提供描述性文本，可用作 alt 文本
            img.alt = `搜索结果 ${index + 1} - ID: ${result.id}`;
             // 基本的图片加载错误处理
            img.onerror = () => {
               img.alt = '图片加载失败';
               img.src = 'placeholder_error.png'; // 可选：指向一个错误占位图片的路径
               // 或者可以显示文本而不是破碎的图片图标
            };


            const indexSpan = document.createElement('span');
            indexSpan.classList.add('index');
            indexSpan.textContent = `#${index}`;

            const descriptionP = document.createElement('p');
            descriptionP.classList.add('description');
            // 使用 ID 和 distance 作为占位描述，因为 API 没有提供像图片那样的文本
            // 如果 API 响应中有更描述性的文本：
            // descriptionP.textContent = result.description || `图片 ID: ${result.id}`;
            // 这里我们暂时用 ID 和 distance 来填充
            descriptionP.textContent = `ID: ${result.id}, 距离: ${result.distance.toFixed(4)}`;


            // 将元素添加到结果项中
            itemDiv.appendChild(img);
            itemDiv.appendChild(indexSpan);
            itemDiv.appendChild(descriptionP);

            // 将结果项添加到网格中
            resultsGrid.appendChild(itemDiv);
        });
    }

    // 处理表单提交事件 (用户按回车)
    searchForm.addEventListener('submit', async (event) => {
        event.preventDefault(); // 阻止页面默认的重新加载行为
        const searchText = searchInput.value.trim(); // 获取并去除首尾空格

        if (!searchText) {
            errorMessage.textContent = '请输入要搜索的文本。';
            resultsInfo.textContent = ''; // 清空信息区
            resultsGrid.innerHTML = ''; // 清空结果区
            return; // 终止执行
        }

        // 显示加载指示器并清除之前的状态
        loadingIndicator.style.display = 'block';
        errorMessage.textContent = '';
        resultsInfo.textContent = '';
        resultsGrid.innerHTML = ''; // 立即清空网格

        try {
            // 发起 POST 请求到 API
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // 指定请求体是 JSON 格式
                },
                body: JSON.stringify({ text: searchText }), // 将搜索文本包装成 JSON
            });

            // 检查响应状态是否成功 (HTTP 状态码 200-299)
            if (!response.ok) {
                 // 尝试从响应体中获取错误详情
                let errorDetails = `HTTP 错误! 状态码: ${response.status}`;
                try {
                    const errorData = await response.json(); // 尝试解析错误响应体
                    errorDetails += ` - ${errorData.message || JSON.stringify(errorData)}`;
                } catch (e) {
                    // 如果响应体不是 JSON 或为空，则忽略
                }
                throw new Error(errorDetails); // 抛出包含状态码和详情的错误
            }

            // 解析 JSON 响应体
            const results = await response.json();
            // 显示结果
            displayResults(results);

        } catch (error) {
            // 捕获 fetch 或 JSON 解析过程中的错误
            console.error('搜索失败:', error);
            resultsInfo.textContent = ''; // 出错时清空信息区
            errorMessage.textContent = `获取结果时出错: ${error.message}`;
            resultsGrid.innerHTML = ''; // 确保出错时网格是空的
        } finally {
            // 无论成功还是失败，最终都隐藏加载指示器
            loadingIndicator.style.display = 'none';
        }
    });

    // 可选: 页面加载时为输入框中的初始值自动触发一次搜索
    // 如果需要此行为，取消下面这行的注释
    // searchForm.dispatchEvent(new Event('submit'));

    // 如果不自动触发，可以清除占位符信息文本
    // resultsInfo.textContent = '输入文本后按回车键进行搜索。';
    // 或者让初始的静态 HTML 内容保持可见，直到第一次搜索发生
});