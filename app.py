import streamlit as st
import pandas as pd
import os
import re
import uuid
import matplotlib.pyplot as plt

# LangChain & Gemini用のライブラリ
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

# --- 環境設定 ---
# StreamlitのSecrets機能や環境変数からGoogle APIキーを設定することを推奨します。
# 例: os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
# ここでは、見つからない場合に備えてサイドバーで入力できるようにします。
if "GOOGLE_API_KEY" not in os.environ:
    st.sidebar.title("🔐 APIキー設定")
    api_key = st.sidebar.text_input("Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.error("Google APIキーを設定してください。")
        st.stop()

# --- カスタムツールの定義 ---
# [cite_start]ドキュメントの`extra_tools` [cite: 95]に対応するツールを定義します。
# ここでは例として、matplotlibでグラフを描画し保存するツールを作成します。

@tool
def create_chart(data_query: str, title: str, xlabel: str, ylabel: str) -> str:
    """
    pandas.DataFrame.plot()を使用してグラフを生成し、ファイルとして保存します。
    Pythonのコード文字列として、df.plot()の呼び出しを受け取ります。
    例: 'df["カラム名"].plot(kind="bar")'
    """
    global df # グローバル変数として定義されたDataFrameを使用
    if 'df' not in globals():
        return "エラー: DataFrameがロードされていません。"
    try:
        # グラフ描画領域を初期化
        plt.figure()

        # 文字列として渡された描画コードを実行
        # 例: data_query = 'df.plot(kind="bar", x="商品", y="売上")'
        # この方法では柔軟性が高いですが、セキュリティリスクも伴います。
        # eval()の使用は慎重に行う必要があります。
        eval(data_query)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()

        # ファイル名が一意になるようにUUIDを生成
        filename = f"chart_{uuid.uuid4()}.png"
        plt.savefig(filename)
        plt.close() # メモリを解放

        return f"グラフが正常に生成され、'{filename}'として保存されました。ユーザーにこのファイル名を伝えてください。"
    except Exception as e:
        return f"グラフの生成中にエラーが発生しました: {e}"

# 作成したツールをリストにまとめる
tools = [create_chart]

# --- Streamlit UIの構築 ---

st.title("📊 AI搭載 データ分析エージェント")
st.markdown("アップロードしたExcel/CSVファイルのデータを、自然言語で分析します。")

uploaded_file = st.file_uploader("分析したいファイルをアップロードしてください (Excel or CSV)", type=["xlsx", "csv"])
user_prompt = st.text_area("どのような分析をしますか？ (例: 商品別の売上を棒グラフで表示して)", height=150)

# グローバル変数としてDataFrameを定義
df = None

if uploaded_file is not None:
    try:
        # ファイルの拡張子に応じて読み込み方法を切り替える
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("プレビュー:", df.head())
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# 分析実行ボタン
if st.button("分析を実行"):
    # ファイルとプロンプトが入力されているかチェック
    if df is None or user_prompt == "":
        st.warning("ファイルがアップロードされていないか、分析内容が入力されていません。")
    else:
        with st.spinner("AIエージェントが分析を実行中です..."):
            try:
                # --- 修正後：正しく安定したLLMのインスタンス化 ---
                # [cite_start]ドキュメントの指示通り、最新のSDKを利用する安定した方法です [cite: 92, 72]。
                # GOOGLE_API_KEY 環境変数は自動的に読み込まれます。
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    temperature=0,
                    convert_system_message_to_human=True
                )
                # -----------------------------------------------

                # create_pandas_dataframe_agent は langchain-experimental に含まれるため、
                # インストールされていればこのまま使用可能です。
                # セキュリティリスクのため、allow_dangerous_code=True の使用には注意が必要です。
                # [cite_start]詳細はレポートのセクション6.1を参照してください [cite: 94, 101]。
                agent = create_pandas_dataframe_agent(
                    llm,
                    df,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    extra_tools=tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    allow_dangerous_code=True # 注意：セキュリティリスクあり
                )

                # エージェントを実行
                response = agent.invoke({"input": user_prompt})

                # --- 結果の表示処理 ---
                st.subheader("分析結果")
                st.write(response["output"])

                # レスポンスの中に画像ファイル名が含まれているか正規表現でチェック
                # エージェントが生成したグラフを表示する
                image_filenames = re.findall(r"chart_[\w-]+\.png", response["output"])
                for filename in image_filenames:
                    if os.path.exists(filename):
                        st.image(filename, caption=f"生成されたグラフ: {filename}")
                    else:
                        st.warning(f"グラフファイル '{filename}' が見つかりませんでした。")
            
            except Exception as e:
                st.error(f"エージェントの実行中に予期せぬエラーが発生しました: {e}")