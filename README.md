# CEPIDL
Correct Exercise Posture Identification is an AI system that analyzes body movements in real time. Using MediaPipe Pose and an LSTM model, it classifies postures as correct, incorrect, or neutral. It includes a repetition counter and provides instant feedback. Designed for web and mobile, it integrates TensorFlow Lite for accuracy.







**DEMO VIDEO BICEP CURLS:**


https://github.com/user-attachments/assets/dd3c59ad-5428-4d30-813d-0350bbf44cd2

Make sure you are visible to camera and lightning must be enough to detect keypoints.As in video can been seen if i am moving my elbow more it will be considered as wrong posture. The cuorrect movement for bicep curls are that our elbows should not be moved and then curls are performed.

this one is the most fastest and easiest solution for correct identification, other solution can also include arcitecture like conv2dlstm, or using separate cnn and then lstm, in this solution we are using mediapipe for our keypoint extraction and then using LSTM to train our sequence.

NOTE: While running on real time on web application delay will be seen. 








**WORKFLOW:**
given in the bicep code file.
1) dataset creation --> you can improve your dataset by adding more data and adjust body landmarks according to your exercise.
2) Model training --> train your model.
3) Real time running --> run on your python environment to test your model.
4) running on web or mobile --> i have provided bicep code for web using fastapi and websocket, make sure to create two files in your front end one for html(client) and one for python(server).









**GRAPHS:**
![image](https://github.com/user-attachments/assets/8e54d955-251c-474e-8d0b-6649d4ab5b35)
![image](https://github.com/user-attachments/assets/bd343d5e-4b75-4bfd-b419-4841ae6bb75f)
![image](https://github.com/user-attachments/assets/b4479d70-e20e-46c9-8f50-050943efe786)






**NOTE:**
This is the whole content provided for bicep curls, 4 other exercises are all being used in this project check my profile for that CEPIDL_2, all 4 exercises content is available there.
