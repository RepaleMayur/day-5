from os import path
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
import face_recognition

mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

mp_selfie_segmentation = mp.solutions.selfie_segmentation

model = mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0)

model_detection = mp_face_detection.FaceDetection()

st.title("Python Project")

add_selectbox = st.sidebar.selectbox(
    "What do you want to do?",
    ("About", "Face Recognition", "Face Detection", "Selfie Segmentation")
)

if add_selectbox == "About":
    st.write("This application only do three operations ")
    st.write("1: Face Recognition")
    st.write("2: Face Detection")
    st.write("3: Selfie Segmentation")

elif add_selectbox == "Face Recognition":
    st.header(add_selectbox)
    image_path = st.sidebar.file_uploader("Upload image")
    image_path1 = st.sidebar.file_uploader("Upload NEW image")
    name = st.sidebar.text_input("Person name in  image1")
    if image_path is not None and image_path1 is not None :
        image1 = np.array(Image.open(image_path))
        st.sidebar.image(image1)
        image2 = np.array(Image.open(image_path1))
        st.sidebar.image(image2)
        
        image1=cv2.resize(image1,(300,400))
        image2=cv2.resize(image2,(300,400))

        image1_face_encoding = face_recognition.face_encodings(image1)[0]
        image1_location = face_recognition.face_locations(image1)[0]
    
        image2_face_encoding = face_recognition.face_encodings(image2)[0]
        
        results = face_recognition.compare_faces([image1_face_encoding], image2_face_encoding)
        
        if results[0] == 1:
            cv2.rectangle(image1, 
                (image1_location[3], image1_location[0]),
                (image1_location[1], image1_location[2]),
                (0, 255, 0),
                2)
            cv2.rectangle(image1, 
                (image1_location[3], image1_location[2] +30), 
                (image1_location[1], image1_location[2]), 
                (0, 255 , 0),
                 cv2.FILLED)
            cv2.putText(image1,name,
                (image1_location[3] + 20, image1_location[2]+20),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (0, 0 , 255),
                1)
            st.write("Both the Images are SAME")
            st.image(image1)
        else:
            st.write("Both the Images are not SAME")
            
elif add_selectbox == "Face Detection":
    st.header(add_selectbox)
    image_path = st.sidebar.file_uploader("Upload image")
    
    if image_path is not None :
        image1 = np.array(Image.open(image_path))
        st.sidebar.image(image1)

        image1=cv2.resize(image1,(300,400))
        
        results1 = model_detection.process(image1)
        
        if results1.detections :
            for landmark in results1.detections:
                print(mp_face_detection.get_key_point(landmark, mp_face_detection.FaceKeyPoint.NOSE_TIP))
                mp_drawing.draw_detection(image1, landmark)\
            
            st.image(image1)
            st.write("Face Detected")
            
        else:
            st.write("No Face Detected")

elif add_selectbox == "Selfie Segmentation":
    st.header(add_selectbox)

    background = st.sidebar.radio("Select colour",('Blue','Green','Background','Background 1'))
    image_path = st.sidebar.file_uploader("Upload image")
    
    if image_path is not None :
        image = np.array(Image.open(image_path))
        st.sidebar.image(image)

        image=cv2.resize(image,(300,400))

        results = model.process(image)
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        
        if background == 'Blue':
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (0, 0 ,255)
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)

        if background == 'Green':
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0)
            bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
            output_image = np.where(condition, image, bg_image)
            st.image(output_image)


        if background == 'Background':
            bg_path = st.sidebar.file_uploader("upload image")
    
            if image_path is None :
                bg_image = np.array(Image.open(bg_image))
                st.sidebar.image(bg_image)

                image=cv2.resize(image,(300,400))
                bg_image=cv2.resize(bg_image,(300,400))


                if bg_image is None :
                    bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                    output_image = np.where(condition, image,bg_image)
                    st.image(output_image)

        else:
            bg_path1 = st.sidebar.file_uploader("Upload new image")
            if image_path is None:
                bg_image = np.array(Image.open(bg_path1))
                st.sidebar.image(bg_image)
                

                image=cv2.resize(image,(300,400))
                bg_image=cv2.resize(bg_image,(300,400))

                if bg_image is None:
                    bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
                    output_image = np.where(condition, image,bg_image)
                    st.image(output_image)