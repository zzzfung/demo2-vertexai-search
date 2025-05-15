document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const textSearchForm = document.getElementById('text-search-form');
    const searchInput = document.getElementById('search-input');
    const resultsInfo = document.getElementById('results-info');
    const resultsGrid = document.getElementById('results-grid');
    const loadingIndicator = document.getElementById('loading-indicator');
    const errorMessage = document.getElementById('error-message');

    // Top image focus area elements
    const imageFocusArea = document.getElementById('image-focus-area');
    const sourceImagePreviewTag = document.getElementById('source-image-preview-tag');
    const closeImageFocusBtn = document.getElementById('close-image-focus-btn');


    // Modal elements
    const openImageModalBtn = document.getElementById('open-image-modal-btn');
    const imageUploadModal = document.getElementById('image-upload-modal');
    const modalCloseBtn = imageUploadModal.querySelector('.modal-close-btn');
    const imageDropArea = document.getElementById('image-drop-area');
    const imageFileInput = document.getElementById('image-file-input'); // The actual file input
    const uploadFileLink = document.getElementById('upload-file-link'); // The <a> tag
    const imageUrlInput = document.getElementById('image-url-input');
    const searchByUrlBtn = document.getElementById('search-by-url-btn');
    const imagePreviewInModal = document.getElementById('image-preview'); // Preview inside modal
    const modalErrorMessage = document.getElementById('modal-error-message');

    // --- API URLs ---
    const TEXT_API_URL = 'https://flask-image-to-text-app-746866758104.us-central1.run.app/search';
    const IMAGE_API_URL = 'https://flask-image-to-text-app-746866758104.us-central1.run.app/imagesearch';

    let lastSearchedImageFile = null;

    function convertGcsToHttps(gcsUrl) {
        if (gcsUrl && gcsUrl.startsWith('gs://')) {
            return gcsUrl.replace('gs://', 'https://storage.googleapis.com/');
        }
        return gcsUrl;
    }

    function displayResults(results) {
        resultsGrid.innerHTML = '';
        resultsInfo.textContent = `检索到 ${results.length} 个结果.`;
        if (results.length === 0) {
            resultsGrid.innerHTML = '<p style="text-align: center; color: #555;">未找到匹配的结果。</p>';
            return;
        }
        results.forEach((result, index) => {
            const imageUrl = convertGcsToHttps(result.gcs_url);
            const itemDiv = document.createElement('div');
            itemDiv.classList.add('result-item');
            const img = document.createElement('img');
            img.src = imageUrl;
            img.alt = `搜索结果 ${index + 1}`;
            img.loading = 'lazy'; // Lazy load images
            img.onerror = () => {
               img.alt = '图片加载失败';
               const errorText = document.createElement('p');
               errorText.style.fontSize = '0.8em'; errorText.style.textAlign = 'center';
               errorText.textContent = '图片加载失败';
               if (img.parentNode) img.parentNode.replaceChild(errorText, img);
            };
            const indexSpan = document.createElement('span');
            indexSpan.classList.add('index');
            indexSpan.textContent = `#${index}`;
            const descriptionP = document.createElement('p');
            descriptionP.classList.add('description');
            descriptionP.innerHTML = `ID: ${result.id}${result.distance !== undefined ? `, <br>距离: ${result.distance.toFixed(4)}` : ''}`;
            itemDiv.appendChild(img);
            itemDiv.appendChild(indexSpan);
            itemDiv.appendChild(descriptionP);
            resultsGrid.appendChild(itemDiv);
        });
    }

    function displaySourceImageInFocusArea(imageFile) {
        if (!imageFile) {
            imageFocusArea.style.display = 'none';
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            sourceImagePreviewTag.src = e.target.result;
            imageFocusArea.style.display = 'block'; // Show the area
        }
        reader.readAsDataURL(imageFile);
    }

    function showLoading(isLoading) {
        loadingIndicator.style.display = isLoading ? 'block' : 'none';
        if (isLoading) {
            resultsInfo.textContent = '';
            resultsGrid.innerHTML = '';
            errorMessage.textContent = '';
            modalErrorMessage.textContent = '';
        }
    }

    function showError(message, isModalError = false) {
        if (isModalError) {
            modalErrorMessage.textContent = message;
        } else {
            errorMessage.textContent = message;
        }
    }

    function clearSearchContext(isTextSearch = false) {
        errorMessage.textContent = '';
        if (isTextSearch || !lastSearchedImageFile) { 
            imageFocusArea.style.display = 'none';
            sourceImagePreviewTag.src = '#';
            // lastSearchedImageFile = null; // Don't nullify here if we want close button to explicitly clear it
        }
    }
    
    if(closeImageFocusBtn) {
        closeImageFocusBtn.addEventListener('click', () => {
            imageFocusArea.style.display = 'none';
            sourceImagePreviewTag.src = '#';
            lastSearchedImageFile = null; 
            // Optionally clear results:
            // resultsInfo.textContent = '';
            // resultsGrid.innerHTML = '';
        });
    }


    // --- Modal Logic ---
    function openModal() {
        imageUploadModal.style.display = 'block';
        modalErrorMessage.textContent = '';
        imagePreviewInModal.style.display = 'none';
        imagePreviewInModal.src = '#';
        imageUrlInput.value = '';
        imageFileInput.value = ''; // Clear file input to allow re-selection of the same file
    }
    function closeModal() { imageUploadModal.style.display = 'none'; }
    openImageModalBtn.addEventListener('click', openModal);
    modalCloseBtn.addEventListener('click', closeModal);
    window.addEventListener('click', (event) => { if (event.target === imageUploadModal) closeModal(); });

    // --- Text Search Logic ---
    textSearchForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        clearSearchContext(true); 

        const searchText = searchInput.value.trim();
        if (!searchText) {
            showError('请输入要搜索的文本。');
            resultsInfo.textContent = ''; resultsGrid.innerHTML = '';
            return;
        }
        showLoading(true);
        try {
            const response = await fetch(TEXT_API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: searchText }),
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({message: "无法解析错误"}));
                throw new Error(`HTTP ${response.status}: ${errorData.message || response.statusText}`);
            }
            const results = await response.json();
            displayResults(results);
        } catch (error) {
            console.error('Text search failed:', error);
            showError(`文本搜索失败: ${error.message}`);
            resultsInfo.textContent = ''; resultsGrid.innerHTML = '';
        } finally { showLoading(false); }
    });

    // --- Image Search Logic ---
    async function performImageSearch(imageFile) {
        if (!imageFile) {
            showError('没有提供图片文件用于搜索。', true);
            return;
        }
        lastSearchedImageFile = imageFile; 
        clearSearchContext(false); 
        showLoading(true);
        closeModal(); 

        const formData = new FormData();
        formData.append('image', imageFile, imageFile.name || 'uploaded_image.jpg');
        try {
            const response = await fetch(IMAGE_API_URL, { method: 'POST', body: formData });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({message: "无法解析错误响应"}));
                throw new Error(`HTTP ${response.status}: ${errorData.message || response.statusText}`);
            }
            const results = await response.json();
            displaySourceImageInFocusArea(lastSearchedImageFile); 
            displayResults(results);
        } catch (error) {
            console.error('Image search failed:', error);
            showError(`图片搜索失败: ${error.message}`);
            resultsInfo.textContent = ''; resultsGrid.innerHTML = '';
            // Only hide focus area if the error is critical for display, 
            // otherwise user might want to see what they tried to search with.
            // imageFocusArea.style.display = 'none'; 
        } finally { showLoading(false); }
    }
    
    // *** 问题1修正处 START ***
    // Handle click on "上传文件" link to trigger file input
    if (uploadFileLink) {
        uploadFileLink.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent default anchor behavior
            imageFileInput.click(); // Programmatically click the hidden file input
        });
    }
    // *** 问题1修正处 END ***

    // Handle file selection from <input type="file">
    imageFileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            if (file.type.startsWith('image/')) {
                performImageSearch(file); 
            } else {
                showError('请选择一个图片文件。', true);
                imageFileInput.value = ''; // Reset file input if invalid file
            }
        }
    });

    // Handle file drop
    imageDropArea.addEventListener('drop', (event) => {
        event.preventDefault(); event.stopPropagation();
        imageDropArea.style.borderColor = '#d0d0d0';
        const file = event.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            performImageSearch(file); 
        } else if (file) {
            showError('请拖放一个图片文件。', true);
        }
    });
    // Drag listeners for styling
    imageDropArea.addEventListener('dragover', (event) => { event.preventDefault(); event.stopPropagation(); imageDropArea.style.borderColor = '#4285f4'; });
    imageDropArea.addEventListener('dragleave', (event) => { event.preventDefault(); event.stopPropagation(); imageDropArea.style.borderColor = '#d0d0d0'; });


    // Image URL Upload
    searchByUrlBtn.addEventListener('click', async () => {
        const url = imageUrlInput.value.trim();
        if (!url) { showError('请输入图片链接。', true); return; }
        modalErrorMessage.textContent = '正在从链接加载图片...';
        imagePreviewInModal.style.display = 'none'; imagePreviewInModal.src = '#';
        try {
            const response = await fetch(url); // Consider adding a timeout or using AbortController
            if (!response.ok) throw new Error(`无法加载图片 (HTTP ${response.status})`);
            const blob = await response.blob();
            if (!blob.type.startsWith('image/')) throw new Error('链接指向的不是有效的图片类型。');
            const fileName = url.substring(url.lastIndexOf('/') + 1).split(/[?#]/)[0] || 'imageFromUrl.jpg';
            const imageFile = new File([blob], fileName, { type: blob.type });
            performImageSearch(imageFile); 
        } catch (error) {
            console.error('Error fetching image from URL:', error);
            showError(`无法从链接加载图片: ${error.message}`, true);
            imagePreviewInModal.style.display = 'none'; // Hide preview on error too
            modalErrorMessage.textContent = `无法从链接加载图片: ${error.message.substring(0, 100)}`; // Clear loading, show error
        }
    });

    clearSearchContext(true); // Initial page state
});