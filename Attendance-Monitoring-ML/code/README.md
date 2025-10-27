# Vision Attendance System - FastAPI Server

A FastAPI-based server for facial recognition attendance management system using Siamese neural networks.

## Features

- Real-time facial recognition using Siamese neural networks
- Face detection using Haar Cascade classifier
- RESTful API endpoints for attendance management
- Support for multiple lab sessions
- Image processing and embedding generation

## Project Structure

```
FastAPI server/
├── main1.py                 # Main FastAPI application
├── verification.py          # Face verification logic
├── embeddingLayer.py        # Neural network embedding layer
├── distanceLayer.py         # Distance calculation layer
├── distance_model.py        # Distance model implementation
├── accuracyMetric.py        # Accuracy metrics
├── tripletLoss.py          # Triplet loss implementation
├── L2norm.py               # L2 normalization
├── preprocessing.py        # Image preprocessing utilities
├── Real_time_detection/    # Detection and image directories
│   ├── return_image/       # Returned processed images
│   ├── input_image/        # Input images for processing
│   └── */                  # Student image directories
└── .gitignore             # Git ignore rules
```

## API Endpoints

### Main Endpoints
- `POST /verify_face` - Verify a face against stored embeddings
- `POST /process_image` - Process and store face images
- `GET /get_returned_image` - Retrieve processed images

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install fastapi uvicorn opencv-python tensorflow numpy
   ```

2. **Download Required Files**
   - Haar Cascade classifier file (excluded from repo)
   - Siamese model files (excluded from repo)

3. **Run the Server**
   ```bash
   uvicorn main1:app --reload --host 0.0.0.0 --port 8000
   ```

## Model Files

The following files are required but excluded from the repository due to size:
- `haarcascade_frontalface_default.xml` - Haar Cascade classifier
- `siamesemodel1.keras` - Siamese neural network model 1
- `siamesemodel2.keras` - Siamese neural network model 2  
- `siamesemodel3.keras` - Siamese neural network model 3

## Configuration

Update the model paths and parameters in the respective Python files according to your setup.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License. 