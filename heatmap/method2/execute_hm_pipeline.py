from hm_pipeline import * 
import pandas as pd 



if __name__ == "__main__":
    dp = DataPreparation( hm_height = 24, hm_width = 24,
                         ip_width = 96, ip_height=96,
                         sigma = 4, stride = 5, noOfkps=9)
    csvPath = r"d:\Shoe_Tryon\kps-datafilter\Datasets\Datasets1to8\FinalDataset_96x96\RGBFinalData.csv"

    heatmaps, images = dp.getKeyPointData(csvPath)

    ims, hms = np.array(images), np.array(heatmaps)
    print("Images shape : ", ims.shape)
    print("Heatmaps shape : ", hms.shape)


    trainmodule = Training()
    train_images, train_heatmap, test_images, test_heatmap = trainmodule.split_data(ims, hms)
    trainmodule.train(ims, hms, epochs=1000, batch_size=32)

    # test = Testing()
    # test.testmodel(r"D:\Shoe_Tryon\Shoe_KP_HM_Training\shoesHM_V1.h5", "testkps")