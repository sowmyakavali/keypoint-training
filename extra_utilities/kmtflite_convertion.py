import tensorflow as tf

def default_convertion(input_model):
    model = tf.keras.models.load_model(input_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(input_model.replace(".h5", "_default.tflite"), "wb").write(tflite_model)
    print("Default convertion Done....✌")

if __name__=="__main__":
    input_model = r"D:\ML-projects\tryon-training\eyewearData\FinalDataset_96x96\eyewear_32b_96s_6kps_2023-08-29MobileNet0.99.h5"
    default_convertion(input_model)
