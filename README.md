# dqn

単純なDQNの実装です。
[このサイト](https://gpa.hateblo.jp/entry/2020/01/18/170714)で紹介しているやつです。

## Install

まず、[PyTorchの公式サイト](https://pytorch.org/)を参考にして、PyTorchをインストールしてください。
「QUICK START LOCALLY」というところを見ればいいです。

その後、以下のコマンドを実行すればインストールできます。

```
$ cd dqn/
$ pip install -e .
```

## Learn

`python train.py`で学習できます。
学習終了時、もしくは`Ctrl-C`等で終了させた際に、学習結果として1回分のプレイ結果を表示します。

### Learning Process

学習経過は、`tensorboard --logdir runs/???`（`???`には学習開始日時が名前に含まれているディレクトリを指定する）を実行すると、

```
TensorBoard 2.1.1 at http://localhost:6006/ (Press CTRL+C to quit)
```

みたいな感じの出力が出てくるので、ブラウザで`http://localhost:6006/`に飛べば見ることができます。
`6006`の部分は変わる場合があるので、出力に従ってください。

飛んでみると、横軸が何エピソード目か、縦軸がそのエピソードで得られた報酬和（0〜200）のグラフが出てくると思います。
左側の「Smoothing」の値を0.99にするといい感じの見た目になります。

## Author

GPA芸人（@\_Yorihime\_W\_）

## License

MIT

