import keras.models
from matplotlib import image
import numpy as np
import cv2



def Init_model():
    
    np.set_printoptions(suppress=True)
    # Load the model
    try:
        model = keras.models.load_model('keras_model.h5')
        data = np.ndarray(shape=(10, 224, 224, 3), dtype=np.float32)
    except:
        print("Model could not be loaded exiting program....")
        exit()
    else:
        return model,data
    

def Get_labels():
    with open('labels.txt','r'):
        

def Det_cam(model,data):
    vid = cv2.VideoCapture(0)

    while(True):
        ret,image=vid.read()
        if not ret:
            break
        
        key=cv2.waitKey(1)
        if key%256==27:
            print("Escaped")
            break
        image=cv2.resize(image,(224,224))
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        # print(prediction)

        for i in prediction:
            if i[0]>0.7:
                text="person"
            if i[1]>0.7:
                text="car"
            if i[2]>0.7:
                text="plane"    
            image=cv2.resize(image,(500,500))
            cv2.putText(image,text,(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2,cv2.LINE_8)
        cv2.imshow("test",image)



def main():
    model,data=Init_model()
    Det_cam(model,data)
if __name__ == '__main__':
    main()
    