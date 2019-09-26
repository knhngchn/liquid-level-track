### 接著跑我的訓練

```
python train.py --dataset csv --csv_train ./annotations.csv --csv_classes ./classes.csv --state_dict ./model/0926_csv_retinanet_8.state_dict --batch_size 4 --dump_prefix 0926_ > log.txt
```


### Train

有切驗證集（建議用這個，才不會和我一樣訓練完沒資料測試XD）：
```
python train.py --dataset csv --csv_train ./annotations.csv --csv_val ./val_annotations.csv --csv_classes ./classes.csv --batch_size 4 --dump_prefix 0926_ > log.txt
```

- 要用這個的話必須先創一個val_annotations.csv，把annotations.csv裡面抽一些出來，放到val_annotations.csv
- 記得抽出來的就不會出現在annotations.csv，然後創好的.csv就放在跟annotations.csv同一層就好
- --dump_prefix給的是存參數的時候名字開頭，假設我設0926_，參數就會存0926_csv_retinanet_{epoch}.state_dict
- 最後面的 > log.txt 只是把training過程的資訊存到log.txt，要監看的話可以再開一個terminal，然後用```tail -f log.txt```

沒切驗證集（全部拿去訓練，不過最後會沒有資料測試模型有沒有over-fitting）：
```
python train.py --dataset csv --csv_train ./annotations.csv --csv_classes ./classes.csv > log.txt
```

### Test

```
python test.py --dataset csv --csv ./val_annotations.csv --csv_classes classes.csv --state_dict /path/to/your/model/weight.state_dict
```

- 平常我是會切三個集，不過2000多張可能只能切兩個集，所以把前面的驗證集拿來測試吧
- 測試的時候每張的recall/precision還有目前的AP都會印在terminal

### Visualize

這是用來可視化預測結果，會把預測的框畫在照片上然後存起來
```
python visualize.py --csv_classes classes.csv --img_prefix ./data/aggregation/ --state_dict /path/to/your/weight.state_dict --step_frame 100 --conf_thres 0.7 --dump_dir ./visualize_image/
```

- 如果有跳出RuntimeError: Error(s) in loading state_dict for ResNet，就在classes.csv裡第二行隨便再加一個類別
- --img_prefix要給的是測試資料的地方，像我資料放在./data/aggregation/底下
- --step_frame要給的是每幾張要拿來測試
- --conf_thres預設是0.85，如果預測結果不好可以調整一下（但別調太低XD）
- --dump_dir要給的是visualize完的照片要存在哪裡

### Versions

- python 3.6.7
- torch 0.4.1

### Others

- 還有很多其他參數可以調整，原則上應該是不用調，想調的話可以看每個.py有哪些parser，然後輸入command的時候照著調整

