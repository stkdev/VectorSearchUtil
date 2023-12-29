import os
import pandas as pd

from vsu.text import VSU_Text_E5
from vsu.image import VSU_Image_CLIP, VSU_Image_EfficientNet


def get_sample_text():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_text.csv"))


def get_sample_image_list():
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_img.csv"))
    df["画像パス"] = df["画像パス"].apply(lambda r: os.path.join(os.path.dirname(__file__), r))
    return df


def sample1_1_simple(df):
    """
    オンメモリDBに格納
    データを入力し、探索する最小限のコード
    """
    vsu = VSU_Text_E5()
    vsu.set_data(df)

    q = "ハトの生態"
    print(f'テキストの近傍探索。クエリ：{q}')
    print(vsu.query_with_info(q))


def sample1_2_add(df):
    """
    オンメモリDBに格納
    データを入力した後にさらにデータを追記する
    　targetの内容が重複しないようにユニーク制約あり
    　
    """
    vsu = VSU_Text_E5()
    df_1 = df.head(15)  # 0~14番目
    df_2 = df.tail(n=20)  # 5~24番目

    # まずdf_2だけ入る
    vsu.set_data(df_2)

    # set_dataをそのまま実行すると一度すべて削除されてから新たにデータが登録される
    # df_1のみが入る
    vsu.set_data(df_1)

    # append=True にすると追記モードとなる
    # データが重複している5~14番目のデータは二重登録されない
    vsu.set_data(df_2, append=True)
    q = "ハトの生態"
    print(f'データの追加後、テキストの近傍探索。クエリ：{q}')
    print(vsu.query_with_info(q))


def sample2_1_simple(df):
    """
    オンメモリDBに格納
    画像パスを入力しベクトル化して保持
    現在は検索の場合も画像パスを指定するよう設計している
    """
    vsu = VSU_Image_CLIP()
    vsu.set_data(df)

    q = get_sample_image_list()["画像パス"][0]
    print(f'画像の近傍探索(CLIP)。クエリ：{q}')
    print(vsu.query_with_info(q))


def sample2_2_zeroshot(df):
    """
    オンメモリDBに格納
    ゼロショット学習により画像からラベルの予測を行う
    set_zeroshot_labels()に候補を指定
    do_zeroshot()によりスコアと結果を返却
    """

    vsu = VSU_Image_CLIP()
    vsu.set_data(df)

    labels = ["a cat", "a dog"]
    vsu.set_zeroshot_labels(labels)
    scores, pred = vsu.do_zeroshot()

    print(f'画像のゼロショット分類。分類キー：{labels}')
    print(pd.DataFrame({"target": vsu.data["target"], "pred": pred}))


def sample3_1_simple(df):
    """
    オンメモリDBに格納
    画像パスを入力しベクトル化して保持
    現在は検索の場合も画像パスを指定するよう設計している
    """
    vsu = VSU_Image_EfficientNet()
    vsu.set_data(df)

    q = get_sample_image_list()["画像パス"][0]
    print(f'画像の近傍探索(EfficientNet)。クエリ：{q}')
    print(vsu.query_with_info(q))


def check():
    # 入力データの仕様
    # 対象列を target、情報として保持しておきたい列を　option1~5 という名前にする
    df = get_sample_text()
    df = df.rename(columns={"タイトル": "target", "作者": "option1"})

    df_img = get_sample_image_list()
    df_img = df_img.rename(columns={"画像パス": "target"})

    sample1_1_simple(df)
    sample1_2_add(df)

    sample2_1_simple(df_img)
    sample2_2_zeroshot(df_img)

    sample3_1_simple(df_img)


if __name__ == '__main__':
    check()
