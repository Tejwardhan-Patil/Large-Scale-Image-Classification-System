import pytest
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Image Classification API"}

# Test health check endpoint
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

# Test classification with a valid image
def test_classification_endpoint():
    files = {"file": ("image.jpg", open("tests/sample_images/test_image.jpg", "rb"), "image/jpeg")}
    response = client.post("/classify", files=files)
    assert response.status_code == 200
    json_response = response.json()
    assert "label" in json_response
    assert "confidence" in json_response
    assert 0 <= json_response["confidence"] <= 1

# Test invalid image format
def test_invalid_image():
    files = {"file": ("invalid_image.txt", open("tests/sample_images/invalid_image.txt", "rb"), "text/plain")}
    response = client.post("/classify", files=files)
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid image format"}

# Test non-existent route (404 error)
def test_404_route():
    response = client.get("/non_existent_route")
    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}

# Test image classification with various images
@pytest.mark.parametrize("image_file, expected_label", [
    ("test_image_1.jpg", "cat"),
    ("test_image_2.jpg", "dog"),
    ("test_image_3.jpg", "car"),
    ("test_image_4.jpg", "bird"),
])
def test_various_images(image_file, expected_label):
    files = {"file": (image_file, open(f"tests/sample_images/{image_file}", "rb"), "image/jpeg")}
    response = client.post("/classify", files=files)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["label"] == expected_label
    assert 0 <= json_response["confidence"] <= 1

# Test large image upload
def test_large_image_upload():
    files = {"file": ("large_image.jpg", open("tests/sample_images/large_image.jpg", "rb"), "image/jpeg")}
    response = client.post("/classify", files=files)
    assert response.status_code == 413  # API has a payload size limit
    assert response.json() == {"detail": "Payload too large"}

# Test for timeouts (performance)
def test_timeout_handling():
    files = {"file": ("slow_image.jpg", open("tests/sample_images/slow_image.jpg", "rb"), "image/jpeg")}
    response = client.post("/classify", files=files, timeout=2)  # Set a low timeout for testing
    assert response.status_code == 504  # Gateway timeout
    assert response.json() == {"detail": "Request timed out"}

# Test invalid method (PUT instead of POST)
def test_invalid_method():
    response = client.put("/classify")
    assert response.status_code == 405  # Method not allowed
    assert response.json() == {"detail": "Method Not Allowed"}

# Test with missing file field
def test_missing_file_field():
    response = client.post("/classify", data={})
    assert response.status_code == 422  # Unprocessable Entity
    assert response.json() == {"detail": "File field is required"}

# Test empty image upload
def test_empty_image_upload():
    files = {"file": ("empty_image.jpg", open("tests/sample_images/empty_image.jpg", "rb"), "image/jpeg")}
    response = client.post("/classify", files=files)
    assert response.status_code == 400
    assert response.json() == {"detail": "Empty image file"}

# Test unauthorized access
def test_unauthorized_access():
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.get("/secure-endpoint", headers=headers)
    assert response.status_code == 401  # Unauthorized
    assert response.json() == {"detail": "Invalid authentication credentials"}

# Test authorized access
def test_authorized_access():
    headers = {"Authorization": "Bearer valid_token"}
    response = client.get("/secure-endpoint", headers=headers)
    assert response.status_code == 200
    assert response.json() == {"message": "Authorized access"}

# Test multiple concurrent requests
@pytest.mark.asyncio
async def test_concurrent_requests():
    files = {"file": ("image.jpg", open("tests/sample_images/test_image.jpg", "rb"), "image/jpeg")}
    import asyncio
    async def send_request():
        response = client.post("/classify", files=files)
        assert response.status_code == 200
        json_response = response.json()
        assert "label" in json_response
        assert "confidence" in json_response
    
    tasks = [send_request() for _ in range(10)]
    await asyncio.gather(*tasks)

# Test malformed JSON input
def test_malformed_json():
    response = client.post("/classify", data="{'invalid': 'json'}", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Malformed JSON"}

# Test allowed file formats
@pytest.mark.parametrize("image_file, content_type", [
    ("test_image.png", "image/png"),
    ("test_image.bmp", "image/bmp"),
    ("test_image.tiff", "image/tiff"),
])
def test_allowed_file_formats(image_file, content_type):
    files = {"file": (image_file, open(f"tests/sample_images/{image_file}", "rb"), content_type)}
    response = client.post("/classify", files=files)
    assert response.status_code == 200
    assert "label" in response.json()

# Test unsupported file format
def test_unsupported_file_format():
    files = {"file": ("test_image.gif", open("tests/sample_images/test_image.gif", "rb"), "image/gif")}
    response = client.post("/classify", files=files)
    assert response.status_code == 415  # Unsupported Media Type
    assert response.json() == {"detail": "Unsupported file format"}

# Test CORS headers
def test_cors_headers():
    response = client.options("/classify")
    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert response.headers["Access-Control-Allow-Origin"] == "*"

# Test rate limiting
def test_rate_limiting():
    files = {"file": ("image.jpg", open("tests/sample_images/test_image.jpg", "rb"), "image/jpeg")}
    
    for _ in range(10):  # Rate limit is 10 requests per minute
        response = client.post("/classify", files=files)
    
    response = client.post("/classify", files=files)
    assert response.status_code == 429  # Too Many Requests
    assert response.json() == {"detail": "Rate limit exceeded"}