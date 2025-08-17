# Krishi Saarthi - Agricultural AI Services

Core ML endpoint for the Krishi Saarthi App, providing VLLM and KCC (Kisan Call Center) AI services for agricultural analysis.

## Project Structure

```
/teamspace/studios/this_studio/
├── main.py                           # Main FastAPI application
├── app/
│   ├── api/                          # API endpoints
│   │   ├── health.py                # Health check endpoints
│   │   └── vllm.py                  # VLLM agricultural response generation endpoints
│   ├── ml-models/                   # Machine Learning models and weights
│   │   └── VLLM/
│   │       └── model_weights/       # Fine-tuned SmolVLM model weights
│   └── services/
│       └── vllm/                    # VLLM service package
│           ├── vllm_service.py      # SmolVLM agricultural response service
│           └── models/              # Model-related code
├── SMOL-vlm-final/                 # Original training directory (can be removed)
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

## Features

### 🌱 VLLM Service
- Fine-tuned SmolVLM for agricultural image analysis
- Agricultural crop disease detection
- Plant health assessment
- Farming recommendations

### 🔍 API Endpoints

**Health Check:**
- `GET /api/health` - Basic health check
- `GET /api/health/detailed` - Detailed service status

**VLLM Agricultural Response Generation:**
- `POST /api/vllm/generate` - Generate responses for agricultural images
- `GET /api/vllm/status` - VLLM service status
- `POST /api/vllm/load` - Manually load VLLM model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

### Agricultural Image Analysis

Upload an agricultural image and ask questions about:
- Crop health and diseases
- Plant growth status
- Farming recommendations
- Pest and disease identification

**Example:**
```python
import requests

# Generate response for agricultural image
with open('crop_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/vllm/generate',
        files={'image': f},
        data={'question': 'What diseases can you identify in these leaves?'}
    )

result = response.json()
print(result['response'])
```

## API Documentation

Once running, visit:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Development

The application uses:
- **FastAPI** for the web framework
- **SmolVLM** fine-tuned model for agricultural analysis
- **PEFT/LoRA** for efficient model inference
- **Transformers** library for model handling

## Model Information

- **Base Model**: HuggingFaceTB/SmolVLM-Base
- **Fine-tuned Model**: Located at `./app/ml-models/VLLM/model_weights`
- **Specialization**: Agricultural image analysis and farming assistance

## Future Enhancements

- **KCC Service**: Kisan Call Center fine-tuned agent integration
- **Batch processing**: Multiple image analysis
- **Real-time streaming**: Live camera feed analysis
- **Mobile optimization**: Lightweight model variants
