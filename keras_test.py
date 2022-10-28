import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


def main():
    (train_generator,val_generator,test_generator)=get_data()
    model=get_model()
    model.fit(
        train_generator,
        epochs=3
    )
    score=model.evaluate(test_generator)
    model.save("Custom_Model\model.h5")
def get_model():
    model=keras.Sequential(
    [
     keras.layers.Flatten(),
     keras.layers.BatchNormalization(epsilon=1e-06,momentum=0.9, weights=None,axis=-1),
     keras.layers.Dense(64,activation="relu"),
    #  keras.layers.Dense(64,activation="selu"),
    #  keras.layers.Dense(32,activation="elu"),
    #  keras.layers.Dense(16,activation="relu"),
     
     keras.layers.Dense(5, activation="softmax")
    ]
    )
    
    model.compile(
        loss="binary_crossentropy",
        optimizer="nadam",
        metrics=['accuracy'])
    return model

def get_data():
    src_path_train= "Dataset/train"
    src_path_test= "Dataset/test"
    
    image_datagen= ImageDataGenerator(
        rescale=1/255.0,
        fill_mode="nearest",
        data_format="channels_last",
        
    )
    
    test_datagen= ImageDataGenerator(
        rescale= 1/255.0,
    )
    
    train_genertor= image_datagen.flow_from_directory(
        directory=src_path_train,
        target_size=(128,128),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        subset="training",
        seed=123
    )
    
    val_generator= image_datagen.flow_from_directory(
        directory=src_path_train,
        target_size=(128,128),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        subset="validation",
        seed=123
    )
    
    test_generator= test_datagen.flow_from_directory(
        directory=src_path_test,
        target_size=(128,128),
        color_mode="rgb",
        batch_size=16,
        class_mode="categorical",
        shuffle=False,
        seed=123
    )
    
    return train_genertor,val_generator,test_generator

if __name__ == '__main__':
    main()