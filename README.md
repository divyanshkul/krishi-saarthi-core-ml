# Krishi Saarthi - Repo for ML Core

Krishi Saarthi is a comprehensive agricultural AI platform that provides intelligent guidance to farmers through specialized machine learning models. The system delivers expert agricultural advice on crop varieties, cultural practices, and visual crop analysis through a unified FastAPI backend.

## Overview

The platform consists of three primary AI services:

1. **KCC Varieties Service** - Provides detailed information about crop varieties, including yield expectations, maturity periods, and growing characteristics, sourced from Kisan Call Centre (KCC) data.
2. **KCC Cultural Practices Service** - Offers guidance on farming practices such as seed rates, sowing methods, irrigation schedules, and fertilizer applications, sourced from Kisan Call Centre (KCC) data.
3. **VLLM Service** - Analyzes agricultural images and provides contextual responses about crop conditions, diseases, and management recommendations

## System Architecture

The system employs a multi-stage processing pipeline that combines domain-specific fine-tuned models with advanced language models for enhanced response generation:

1. **Primary Response Generation** - Fine-tuned TinyLlama models generate initial agricultural responses
2. **Information Extraction** - Structured parsing of key agricultural data points
3. **Knowledge Verification** - Google Gemini integration for fact-checking and enhancement
4. **Response Synthesis** - Final compilation of verified, actionable agricultural guidance

## Models and Training

| Service                | Base Model                         | Fine-tuning Approach | Training Focus                                            | Model Size      |
| ---------------------- | ---------------------------------- | -------------------- | --------------------------------------------------------- | --------------- |
| KCC Varieties          | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Prompt Tuning (PEFT) | Soybean variety information, yield data, maturity periods | 1.1B parameters |
| KCC Cultural Practices | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Prompt Tuning (PEFT) | Farming practices, seed rates, sowing methods, timing     | 1.1B parameters |
| VLLM                   | HuggingFaceTB/SmolVLM-Base         | LoRA Adapters (PEFT) | Agricultural image analysis, crop assessment              | Multi-modal     |
| Enhancement            | Google Gemini 2.5 Flash            | API Integration      | Response verification and synthesis                       | Cloud-based     |

## Machine Learning Models

1. Fine-tuned SmolVLM using QLoRA
   <img width="1794" height="813" alt="IMG_5999" src="https://github.com/user-attachments/assets/e950e665-a89b-49b9-b7ad-ed868df4ff31" />

2. Prompt-tuned TinyLLaMA + Grounding via Vertex AI Google Search
   <img width="3774" height="1585" alt="Krishi Saarthi - Model Prompt Tuning - The Misfits" src="https://github.com/user-attachments/assets/38fde04b-1e64-436a-addd-517a4616817d" />

### Training Methodology

**TinyLlama Models (KCC Services)**

- **Technique**: Parameter-Efficient Fine-Tuning (PEFT) using prompt tuning
- **Dataset**: Agricultural knowledge base covering Indian farming conditions
- **Specialization**: Domain-specific prompt initialization for agricultural contexts
- **Optimization**: Focused on practical farming advice for Central Indian conditions

**SmolVLM (VLLM Service)**

- **Technique**: Low-Rank Adaptation (LoRA) for vision-language tasks
- **Dataset**: Agricultural image-text pairs covering crop analysis scenarios
- **Quantization**: 4-bit quantization with BitsAndBytesConfig for efficient inference
- **Specialization**: Fine-tuned for agricultural visual understanding and response generation

## System Requirements

### Hardware Requirements

- **NVIDIA GPU**: CUDA-compatible GPU required for model inference
- **VRAM**: Minimum 8GB GPU memory recommended
- **RAM**: 16GB system memory minimum
- **Storage**: 50GB available space for models and dependencies

### Software Requirements

- Python 3.12
- CUDA Toolkit compatible with PyTorch
- Docker (optional for containerized deployment)

## API Endpoints

### KCC Varieties Service

- `POST /api/kcc/varieties/query` - Process variety-specific queries
- `GET /api/kcc/varieties/status` - Service health status
- `POST /api/kcc/varieties/load` - Manual model loading

### KCC Cultural Practices Service

- `POST /api/kcc/cultural/query` - Process farming practice queries
- `GET /api/kcc/cultural/status` - Service health status
- `POST /api/kcc/cultural/load` - Manual model loading

### VLLM Service

- `POST /api/vllm/generate` - Analyze agricultural images with questions
- `GET /api/vllm/status` - Service health status
- `POST /api/vllm/load` - Manual model loading

### System Health

- `GET /api/health` - Basic system health check
- `GET /api/health/detailed` - Comprehensive service status

## Installation and Setup

### Prerequisites

Ensure NVIDIA GPU drivers and CUDA toolkit are properly installed.

### Environment Configuration

1. Copy configuration templates:

   ```bash
   cp .env.example .env
   cp firebase.json.example krishi-saarthi-main-[your-key-id].json
   ```

2. Configure environment variables in `.env`:

   ```bash
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

3. Update Firebase service account credentials in the JSON file.

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

### Docker Deployment

```bash
# Build the image
docker build -t krishi-saarthi .

# Run the container
docker run -p 8000:8000 --gpus all krishi-saarthi
```

## Model Weights Structure

```
app/ml-models/
├── KCC/
│   ├── varieties_weights/          # TinyLlama variety model weights
│   └── cultural_practices_weights/ # TinyLlama cultural practices weights
└── VLLM/
    └── model_weights/              # SmolVLM vision-language model weights
```

## Performance Characteristics

The system is optimized for agricultural use cases with the following performance profile:

- **Response Time**: 15-45 seconds per query (includes multi-stage processing)
- **Concurrent Requests**: Supports multiple simultaneous queries
- **Memory Efficiency**: Quantized models reduce GPU memory requirements
- **Accuracy**: Domain-specific fine-tuning provides relevant agricultural guidance

## Use Cases

**Farmer Assistance**

- Crop variety selection based on local conditions
- Optimal sowing practices and timing recommendations
- Visual crop health assessment and disease identification
- Fertilizer application guidance and scheduling

**Agricultural Extension Services**

- Standardized agricultural advice delivery
- Scalable farmer education and support
- Regional farming practice recommendations
- Crop management decision support

## Technical Implementation Details

The platform uses a robust service-oriented architecture with:

- **FastAPI Framework** for high-performance API services
- **Asynchronous Processing** for efficient request handling
- **PEFT Integration** for memory-efficient model fine-tuning
- **Multi-modal Support** combining text and image processing capabilities
- **Cloud Integration** with Google services for enhanced responses
