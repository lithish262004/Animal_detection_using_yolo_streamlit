# Animal_detection_using_yolo_streamlit


Project Description: Animal Detection Using YOLO
Overview
This project implements an animal detection application using the YOLO (You Only Look Once) deep learning model, designed to identify various animal species in real-time through webcam input. Built with Streamlit, the application provides an intuitive user interface that allows users to observe live animal detection, along with descriptive information about each identified species.

Real-Time Detection: Utilizes a webcam feed to capture images and detect animals in real-time, providing immediate feedback to users.
Custom Model: Employs a trained YOLOv5 model (best.pt) specifically fine-tuned for detecting a range of animals, including mammals, birds, and reptiles.
User-Friendly Interface: Built with Streamlit, the application features a clean and visually appealing design, enhancing the user experience.
Animal Descriptions: For each detected animal, the application displays a brief description, providing educational context about the species.

working:
Webcam Access: The application accesses the user's webcam to start capturing video.
Frame Processing: Each frame from the webcam feed is processed by the YOLO model to detect animals.
Bounding Box Drawing: Detected animals are highlighted with bounding boxes and confidence scores are displayed on the video feed.
Description Display: For each identified animal, a corresponding description is fetched from a predefined dictionary and displayed on the interface.
Real-Time Interaction: Users can observe the detection process in real-time, with the ability to stop the feed using a keypress
