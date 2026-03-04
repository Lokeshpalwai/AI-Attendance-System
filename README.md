# AI Attendance System

An AI-based attendance system that detects and recognizes student faces using OpenCV and Python.

## Features

* Face detection using Haar Cascade
* Face recognition using LBPH algorithm
* Automatic attendance recording
* Daily attendance reports
* Student database using CSV

## Technologies Used

* Python
* OpenCV
* NumPy
* Computer Vision

## Project Structure

```
AI-Attendance-System
│
├── dataset
├── trainer
├── capture_images.py
├── train_model.py
├── attendance.py
├── students.csv
├── requirements.txt
└── README.md
```

## Demo

![Attendance System](screenshots/attendance_demo.png)

## How to Run

1. Install dependencies

```
pip install -r requirements.txt
```

2. Capture student images

```
python capture_images.py
```

3. Train the model

```
python train_model.py
```

4. Start attendance system

```
python attendance.py
```

## Output

The system detects student faces and automatically records attendance with timestamp.

## Future Improvements

* Web dashboard
* Database integration
* Cloud deployment
# AI-Attendance-System
