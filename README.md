# bitcoinの価格推移予測

scikit-learnを使用し、アルゴリズムは決定木(DecisionTreeClassifier)。  
データはここから利用し(https://www.kaggle.com/mczielinski/bitcoin-historical-data )、`coincheckJPY`のデータを使用する。

## 結果
```
[mute@localhost bitcoin-prediction-py (master)]$ python main.py
SGDClassifier 0.494915254237
DecisionTreeClassifier 0.583050847458
SVM 0.583050847458
```
