# Vehicle_Lincense_Plate_Recognition

> Data Distribution:

|   components    |               number                |
| :-------------: | :---------------------------------: |
|    integers     |                4679                 |
|    alphabets    |                9796                 |
| Chinese_letters |                3974                 |
|   percentage    | training_set : testing_set == 4 : 1 |

> Usage:
1. To see the data_set accuracy: `python restore_lenet.py`, and change the parameters of the test_lst in main function.

2. To predict the plate of a image containing part of a car including its vehicle-plate: `python predict_plate.py SRC_IMAGE`.

3. Any problems on displaying Chinese characters on matplotlib windows, please refer to [this](http://www.cnblogs.com/ZhengPeng7/p/8823956.html).

> In saved model:

| matches  | Validation Accuracy |
| :------: | :-----------------: |
| 字母+数字+汉字 |       98.98%        |
|  字母+数字   |       99.35%        |
|    汉字    |       97.48%        |
|    字母    |       99.40%        |
|    数字    |       99.27%        |


+ Extract Characters:
  - Effects:
    + ![ori](./images/cars/car_0.jpg)

    + ![edge](./images/cars/recognition/edge_car_0.png)

    + &emsp;1. Canny  
      &emsp;2. Denoise  
      &emsp;3. Morphology  
      &emsp;4. Find\_contours  
      &emsp;5. get\_rects  
      &emsp;6. Select\_the\_very\_rect  
      &emsp;7. Cut\_out\_the\_plate\_area  

    + ![plate](./images/plate.png)

    + ![characters](./images/cars/recognition/characters_car_0.png)

    + &emsp;8. Border\_denoise  
      &emsp;9. Thresholding  
      &emsp;10. Denoise  
      &emsp;11. Kick\_out\_white\_circle\_dot  
      &emsp;12. Get\_vertical\_split\_lines  
      &emsp;13. Split_characters

    + ![su](./images/苏.png) ![A](./images/A.png) ![0](./images/0.png) ![C](./images/C.png) ![5](./images/5.png) ![6](./images/6.png)
+ Character Recognition:
  - Data Transformation:
      + img <--> tfrecords -> array
  - Construct Lenet-5
  - Restore Lenet-5
      > In saved model:

      | matches  | Validation Accuracy |
      | :------: | :-----------------: |
      | 字母+数字+汉字 |       98.98%        |
      |  字母+数字   |       99.35%        |
      |    汉字    |       97.48%        |
      |    字母    |       99.40%        |
      |    数字    |       99.27%        |
      &emsp;&emsp;**Accuracies on training set and test set**
      ![ACC](./images/Acc_in_training_on_alp_int_lett.png)
      &emsp;&emsp;**Recall Rates on Alphabet, integers and Chinese_letters**
      ![alp_int_lett](./images/Recall_rate_in_test_on_alp_int_lett.png)
      &emsp;&emsp;**Recall Rates on Alphabets and Integers**
      ![alp_int](./images/Recall_rate_in_test_on_alp_int.png)
      &emsp;&emsp;**Recall Rates on Chinese_letters**
      ![ChineseLetters](./images/Recall_rate_in_test_on_ChineseLetters.png)
      &emsp;&emsp;**Recall Rates on Alphabet**
      ![alphabets](./images/Recall_rate_in_test_on_alphabets.png)
      &emsp;&emsp;**Recall Rates on Integers**
      ![integers](./images/Recall_rate_in_test_on_integers.png)
+ Combination:
  - ![show](./images/cars/recognition/Recognition_car_0.png)
