# 👤 Face Recognition Attendance System

An automated attendance system using real-time face recognition with Siamese Neural Networks, PostgreSQL for data storage, and a user-friendly frontend dashboard.

## 📌 Overview

This project automates the attendance tracking process by leveraging facial recognition technology. It captures facial images via a webcam, verifies identity using a Siamese Neural Network, logs attendance in a PostgreSQL database, and displays records on a responsive frontend interface.

## 🛠️ Technical Approach

### 🎥 1. Image Capture and Face Detection

- **OpenCV**: Real-time facial image capture from connected webcam
- **MTCNN (Multi-task Cascaded Convolutional Networks)**: Robust face detection handling variations in:
  - 💡 Lighting conditions
  - 🔄 Facial poses
  - 🎭 Partial occlusions

### 🧠 2. Face Verification with Siamese Neural Network

- 📊 Generates fixed-length feature vectors from facial images
- 🔍 Compares input face with registered students using distance metrics
- ⚙️ Configurable similarity threshold for verification
- 📐 Euclidean distance calculation for face matching

### 💾 3. Database Integration

- **PostgreSQL**: Secure, structured storage of:
  - 👥 Student profiles
  - 📝 Attendance records
  - ⏰ Timestamped entries
- 🗄️ Relational database design for efficient querying

### 🖥️ 4. Frontend Dashboard

- 📱 Responsive web interface for attendance management
- ⚡ Real-time attendance status display
- 👨‍💼 Administrative functions for teachers
- 📊 Data visualization of attendance patterns

## ✨ Key Features

✅ Real-time face detection and recognition  
✅ High accuracy with Siamese Network embeddings  
✅ Secure database storage with PostgreSQL  
✅ User-friendly web interface for attendance management  
✅ Scalable solution for classroom/enterprise use  
✅ Configurable similarity thresholds

## 📸 Screenshots

### 🔐Login System

![Face Detection](/img/Login.jpg)  
\_Login_for_the_client_side

### 👨‍💻Program flow

![Dashboard](/img/code.png)  
_Administrative view of attendance records_

### ✅ Student Verification

![Verification](/img/1.jpg)  
_Successful student identification process_

### 📑 Database: PostegreSQL

![Verification](/img/database.png)  
_Successful student identification process_

## 🚀 Future Enhancements

- 📱 Mobile app integration
- 👥 Multi-face simultaneous detection
- 🛡️ Liveness detection to prevent spoofing
- 📈 Advanced analytics and reporting
- ☁️ Cloud deployment for remote access

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
