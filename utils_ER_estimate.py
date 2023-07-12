import numpy as np
import mediapipe as mp
from PIL import Image
import io
import os


# 画像をバイナリとndarrayで返します
def get_image_PIL(path="test_img.jpg"):
    img = Image.open(path)
    img_ndarray = np.array(img)
    # BYTES
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    return img_ndarray, img_bytes

# 拍手とグッドサインを識別します
def detect_labels(client, image_binary):

    response = client.detect_labels(Image={"Bytes": image_binary},
        MaxLabels=100, MinConfidence=45)
    
    for label in response['Labels']:
        if "Applause" in label['Name']:
            return "CLAP"
        if "Thumbs Up" in label['Name']:
            return "THUMBS"
        
    return "JOIN"

# 感情を推定します
def detect_faces_and_emotions(client, image_binary):

    rekognition_response = client.detect_faces(
        Image={"Bytes": image_binary}, Attributes=["ALL"]
    )

    face_emotion = "HAPPY"
    for item in rekognition_response.get("FaceDetails"):
        
        for emotion in item.get("Emotions"):
            face_emotion = emotion.get("Type")
            break
                
    return face_emotion

# 骨格データから様々なリアクションを推定します
def detect_reaction(client, img_ndarray, img_bytes):

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        results = pose.process(img_ndarray)
            
        d={}
        if results.pose_landmarks is None:
            reaction = "LEAVE"
        else:
            reaction = detect_labels(client, img_bytes)
            if reaction == "JOIN":
                mouth_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x
                mouth_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y
                sholder_r_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
                sholder_r_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
                sholder_l_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
                sholder_l_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
                elbow_r_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
                elbow_r_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
                elbow_l_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
                elbow_l_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
                index_r_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x
                index_r_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y
                index_l_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x
                index_l_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y


                # RAISE_HAND
                if (mouth_y >= index_r_y and mouth_y < index_l_y) or (mouth_y < index_r_y and mouth_y >= index_l_y):
                    reaction = "RAISE_HAND"
                # AGREE
                elif abs(elbow_l_x - sholder_l_x) > 0.05 and \
                    abs(elbow_r_x - sholder_r_x) > 0.05 and \
                    abs(index_l_x - index_r_x) < 0.2 and \
                    sholder_l_y > index_l_y and sholder_r_y > index_r_y:
                    reaction = "AGREE"
        return reaction



def ER_estimate(client, img_ndarray, img_bytes):
    reaction = detect_reaction(client, img_ndarray, img_bytes)

    if reaction == "LEAVE":
        face_emotion = "HAPPY"
    else:
        face_emotion = detect_faces_and_emotions(client, img_bytes)
    
    print(reaction, face_emotion)
    return face_emotion, reaction


if __name__ == "__main__":

    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    key_id = os.getenv("EMOTION_KEY")
    access_key = os.getenv("EMOTION_ACCESS")
    rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=key_id,
    aws_secret_access_key=access_key,
    region_name="us-east-1",
    )
    img_list = [f"img/cp{i}.jpg" for i in range(1,4)]
    for img in img_list:
        img_ndarray, img_bytes = get_image_PIL(img)
        reaction, face_emotion = ER_estimate(rekognition, img_ndarray, img_bytes)