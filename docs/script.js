document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const previewImg = document.getElementById('previewImg');
    const imagePreview = document.getElementById('imagePreview');
    const predictBtn = document.getElementById('predictBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const uploadForm = document.getElementById('uploadForm');

    // Enable/disable predict button based on file selection
    function updatePredictButton() {
        predictBtn.disabled = !fileInput.files.length;
    }

    // Show image preview
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            imagePreview.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    // Handle file selection
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            showPreview(fileInput.files[0]);
            updatePredictButton();
        }
    });

    // Click to trigger file input
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            showPreview(files[0]);
            updatePredictButton();
        }
    });

    // Form submission via AJAX
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        predictBtn.disabled = true;
        loadingSpinner.style.display = 'block';

        const formData = new FormData(uploadForm);
        try {
            const response = await fetch('/', {
                method: 'POST',
                body: formData
            });
            const result = await response.text();
            // Update page with response
            const parser = new DOMParser();
            const doc = parser.parseFromString(result, 'text/html');
            const newBody = doc.querySelector('body').innerHTML;
            document.body.innerHTML = newBody;
            // Reattach event listeners
            reinitializeScripts();
        } catch (error) {
            loadingSpinner.style.display = 'none';
            predictBtn.disabled = false;
            alert('Error: ' + error.message);
        }
    });

    // Reattach event listeners after AJAX
    function reinitializeScripts() {
        const newUploadArea = document.getElementById('uploadArea');
        const newFileInput = document.getElementById('fileInput');
        const newPreviewImg = document.getElementById('previewImg');
        const newImagePreview = document.getElementById('imagePreview');
        const newPredictBtn = document.getElementById('predictBtn');
        const newLoadingSpinner = document.getElementById('loadingSpinner');
        const newUploadForm = document.getElementById('uploadForm');

        newFileInput.addEventListener('change', () => {
            if (newFileInput.files.length > 0) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    newPreviewImg.src = e.target.result;
                    newImagePreview.style.display = 'block';
                };
                reader.readAsDataURL(newFileInput.files[0]);
                newPredictBtn.disabled = false;
            }
        });

        newUploadArea.addEventListener('click', () => {
            newFileInput.click();
        });

        newUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            newUploadArea.classList.add('dragover');
        });

        newUploadArea.addEventListener('dragleave', () => {
            newUploadArea.classList.remove('dragover');
        });

        newUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            newUploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                newFileInput.files = files;
                const reader = new FileReader();
                reader.onload = (e) => {
                    newPreviewImg.src = e.target.result;
                    newImagePreview.style.display = 'block';
                };
                reader.readAsDataURL(files[0]);
                newPredictBtn.disabled = false;
            }
        });

        newUploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            newPredictBtn.disabled = true;
            newLoadingSpinner.style.display = 'block';
            const formData = new FormData(newUploadForm);
            try {
                const response = await fetch('/', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.text();
                const parser = new DOMParser();
                const doc = parser.parseFromString(result, 'text/html');
                const newBody = doc.querySelector('body').innerHTML;
                document.body.innerHTML = newBody;
                reinitializeScripts();
            } catch (error) {
                newLoadingSpinner.style.display = 'none';
                newPredictBtn.disabled = false;
                alert('Error: ' + error.message);
            }
        });
    }
});