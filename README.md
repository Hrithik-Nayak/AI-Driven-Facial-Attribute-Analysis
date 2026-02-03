https://drive.google.com/drive/folders/1oKgSYnDKTtlovW3erNK6RI9puN5w7uqH?usp=sharing





import cv2



def faceBox(faceNet,frame):
    print(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    blob = cv2.dnn.blobFromImage(frame,1.0,(227,227),[104,117,123],swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence > 0.8:
            x1 = int(detection[0,0,i,3] * frameWidth)
            y1 = int(detection[0,0,i,4] * frameHeight)
            x2 = int(detection[0,0,i,5] * frameWidth)
            y2 = int(detection[0,0,i,6] * frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
    return frame, bboxs




faceProto = "opencv_face_detector.pbtxt"
faceMOdel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"



faceNet = cv2.dnn.readNet(faceMOdel,faceProto)
ageNet = cv2.dnn.readNet(ageModel,ageProto)
genderNet = cv2.dnn.readNet(genderModel,genderProto)


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
# ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-25)', '(26-30)', '(31-35)','(36-40)', '(41-45)', '(46-50)', '(51-55)', '(56-60)', '(61-65)', '(66-70)', '(71-75)', '(76-80)', '(81-85)', '(86-90)', '(91-95)', '(96-100)']
ageList = [('2-17'),('18-20'),('21-22'), ('23-24'), ('25-26'), ('27-28'), ('29-30'),
('31-32'), ('33-34'), ('35-36'), ('37-38'), ('39-40'),
('41-42'), ('43-44'), ('45-46'), ('47-48'), ('49-50'),
('51-52'), ('53-54'), ('55-56'), ('57-58'), ('59-60'),
('61-62'), ('63-64'), ('65-66'), ('67-68'), ('69-70'),
('71-72'), ('73-74'), ('75-76'), ('77-78'), ('79-80'),
('81-82'), ('83-84'), ('85-86'), ('87-88'), ('89-90'),
('91-92'), ('93-94'), ('95-96'), ('97-98'), ('99-100')
]
genderList = ['Male','Female']

video = cv2.VideoCapture(0)

while True:
    ret,frame=video.read()
    frame, bboxs = faceBox(faceNet,frame)
    for bbox in bboxs:
        # face = frame[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-10):min(bbox[3]+10,frame.shape[0]-1),max(0,bbox[0]-10):min(bbox[2]+10,frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = "{}, {}".format(gender, age)
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,255,0),-1)    
        cv2.putText(frame,label,(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)


    cv2.imshow("Age-Gender",frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
