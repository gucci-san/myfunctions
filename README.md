## 流用できる関数とかクラスを持っておくリポジトリ

### テーブルデータ
* df_with_all_feature = pd.read(train.feather).append(pd.read(test.feahter))