# ColAE

## 

## 概要
- ColAEは、Hypersepctralの次元削減手法である。
  - 前処理
    - superpixel segmentation: 画像を類似度に基づくpixelの集合体で分割
  - 本処理
    - Auto-Encoderによる学習
    - 各superpixelごとのAuto-Encoderによる再構成誤差に対して、次元削減後に各superpixel間の構造を保つための多様体損失を加えて、損失を計算する。
   
  
- 以下の論文で提案されている。

  A Collaborative Superpixelwise Autoencoder for Unsupervised Dimension Reduction in Hyperspectral Images [Yao+ (2023)]
  
  DOI: [https://www.mdpi.com/2072-4292/15/17/4211]

- その次元削減手法を論文をもとに実装したものがこのリポジトリにあるファイルである。

![論文紹介資料(日本語)](ThesisReview_jp.pdf)

### Datasets
#### HISUI （Hyperspectral Imager SUIte）
- Port
- Tokyo
#### M^3 (Moon Mineralogy Mapper)
- Schrodinger Creater
