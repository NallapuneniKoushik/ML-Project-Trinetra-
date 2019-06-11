# -*- coding: utf-8 -*-
"""


"""
import face_recognition as fr
import cv2
import pickle
import numpy as np
 
def addface() :
    captured_img = []
    s = int(input("1. From webcam\n2. From photo\nEnter selection : "))
    if(s==1) :
        print("**************************")
        print("     Loading webcam")
        print("**************************")
        
        print("\nEnter q to capture a image")
        vid = cv2.VideoCapture(0)
        
    
        while True : 
            
            _, unknown_image = vid.read()
               
            face_locations = fr.face_locations(unknown_image)
            #face_encodings = fr.face_encodings(unknown_image, face_locations)
            
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (255,0,0)
            lineType               = 2
            count = 1
            
            for (top,right,bottom,left) in face_locations :
                cv2.rectangle(unknown_image,(left,top),(right,bottom),(255,0,0),2)
                cv2.putText(unknown_image,str(count), (left,top - 10), font, fontScale, fontColor, lineType)    
                count = count + 1
                captured_img = unknown_image
                
            cv2.imshow('output',unknown_image)           
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        vid.release()
        cv2.destroyAllWindows()
        print("\nCaptured!")

    if(s==2) :
        print("**************************")
        photo_path = input("Enter the path of the file : ")
        captured_img = cv2.imread(photo_path)
        face_locations = fr.face_locations(captured_img)
        #face_encodings = fr.face_encodings(unknown_image, face_locations)
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2
        count = 1
        
        for (top,right,bottom,left) in face_locations :
            cv2.rectangle(captured_img,(left,top),(right,bottom),(255,0,0),2)
            cv2.putText(captured_img,str(count), (left,top - 10), font, fontScale, fontColor, lineType)    
            count = count + 1
            
    cv2.imshow("capture", captured_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    name = input("\nEnter the name of the person : ")
    
    face_locations = fr.face_locations(captured_img)
    face_encodings = [fr.face_encodings(captured_img, face_locations)[0]]
    pickle_in= open("encodings.pickle", "rb")
    encodings = pickle.load(pickle_in)
    encodings = encodings + face_encodings
    pickle_in.close()
    
    pickle_in= open("names.pickle", "rb")
    names = pickle.load(pickle_in)
    names = names + [name]
    print(names)
    pickle_in.close()
    
    pickle_out = open("encodings.pickle", "wb")
    pickle.dump(encodings, pickle_out)
    pickle_out.close()
    
    pickle_out = open("names.pickle", "wb")
    pickle.dump(names, pickle_out)
    pickle_out.close()
    
    print("\n Successfully added " + str(name))

def detectface() :
    
    hog = cv2.HOGDescriptor()
    pickle_in= open("names.pickle", "rb")
    people = pickle.load(pickle_in)    
    pickle_in.close()
    
    pickle_in= open("encodings.pickle", "rb")
    known_encodings = pickle.load(pickle_in)
    pickle_in.close()
    known_encodings = np.asarray(known_encodings)
    
    vid = cv2.VideoCapture(0)

    while True : 
            
        _, unknown_image = vid.read()
           
        face_locations = fr.face_locations(unknown_image)
        face_encodings = fr.face_encodings(unknown_image, face_locations)
        
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,0,0)
        lineType               = 2
        
        for (top,right,bottom,left), fe in zip(face_locations,face_encodings):
            cv2.rectangle(unknown_image,(left,top),(right,bottom),(255,0,0),2)
            matches = fr.compare_faces(known_encodings, fe, tolerance = 0.49)
        
            name = "Unknown"
        
            # If a match was found in known_face_encodings, just use the first one.
            if True in matches:
                first_match_index = matches.index(True)
                name = people[first_match_index]
            
            cv2.putText(unknown_image, name, (left,top - 10), font, fontScale, fontColor, lineType)    
            
        cv2.imshow('output',unknown_image)           
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

try :
    while True :
        s = int(input("1. Add faces \n2. Detect faces\n3. Exit \nEnter selection : "))
        
        if(s==1) :
            addface()
        
        elif(s==2) :
            detectface()
            
        elif(s==3) :
            print("Exiting")
            break
        
except KeyboardInterrupt :
    print("\nStopping program")
