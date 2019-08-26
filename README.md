# food-classification  
# Usage  
## Step1. データセットの準備  
https://www.vision.ee.ethz.ch/datasets_extra/food-101/  
../CNN_ver1 においてください  

## Step2. 実行方法  
$ python train_test.py –mode [train or test]  

※テストモードはトレーニングモードの後に実行してください  

# 結果
## AlexNet augmentationなし  
loss: 10.7118 - acc: 0.3354  
lossが下がらない

## ZFNet augmentationなし  
loss: 1.0899 - acc: 0.4141  
lossが下がらない  

## CNN_1 augmentationなし Flatten使用  
loss: 1.1489 - acc: 0.5733  
training errorは綺麗に収束していたが、validation errorが上記のところから上がってしまっていて、過学習の傾向にある。  

## CNN_1 augmentationなし Global Average Pooling使用  
loss: 0.7896 - acc: 0.6026  
flattenでパラメータが増えすぎていたので、パラメータを減らせるGlobal Average Poolingを採用する。  
過学習は抑えれたみたいだけど、まだ精度が低い。  

## CNN_1 augmentationあり Flatten使用  
loss: 1.1530 - acc: 0.5912  

## CNN_1 augmentationあり Global Average Pooling使用  
loss: 0.064 - acc: 0.6415  

## CNN  augmentationなし  
loss: 0.045 - acc: 0.8151  
Dropoutをかませつつ過学習を抑えて、層を深くしてみた。  

## CNN  augmentationあり  
loss: 0.033 - acc: 0.8535  
