# 衛星画像の時間軸補間
- 気象衛星から得られる日射量情報に対して，雲の存在をモデル化する事で時間軸の補完を試みる
  - model : uGMM, uEpanechnikovMMによるモデル
  - Exp1_component : コンポネントがGaussianとEpanechnikovの場合の比較
  - Exp2_estimate : 選んだフレーム中，真ん中のフレームで混合モデルの静的なパラメータを先に推定してから前後で移動のパラメータだけ推定した場合と，３枚使って全部推定した場合（中心の位置は先にGauusianで推定？）の比較
  - Exp3_stride : 結合するフレームを重ねる場合と重ねない場合の比較．動画が自然に出来上がるか？
  - misc : その他のファイル