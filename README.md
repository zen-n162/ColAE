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
1. HISUI (Hyperspectral Imager SUIte）
- 空間分解： 20~31 m/pixel
- 観測幅： 20 km
- Spectral
  - チャネル数： **185** (VNIR:57, SWIR:128)
  - 観測波長帯： **0.4-2.5 μm**
  - 波長分解能： VNIR 10 nm, SWIR 12.5 nm
##### データ
- **Port** (181 channel)
  - Vegetation / Opne water / Land use
- **Tokyo** (430 nmを除く180 channel)
  - Vegetation / Open water / others
  
2. $M^3$ (Moon Mineralogy Mapper)
- 空間分解： 125 m/pixel
- Spectral
  - チャネル数： **85** (83)
  - 観測波長帯： **460-2976 nm** (540.84~2976.20 nmを使用)
##### データ
- **Schrodinger Creater** (120x120 km, S74.5°E137.0°)
