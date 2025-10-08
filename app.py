# 1. 必要なライブラリをすべてインポート
import streamlit as st
import pandas as pd
import os
import re
import uuid
import matplotlib.pyplot as plt
from langchain.tools import tool
# Gemini用のライブラリをインポート
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# ----------------------------------------------------------------------
# StreamlitアプリのUI部分
# ----------------------------------------------------------------------

st.title("📄 AIによるデータ分析レポートツール")

# --- サイドバーでのAPIキー設定 ---
st.sidebar.header("APIキー設定")

# APIキーの入力 (Gemini用に修正)
api_key_input = st.sidebar.text_input(
    "Google AI APIキー",
    type="password",
    placeholder="APIキーをここに入力してください"
)

# 環境変数にAPIキーを設定 (Gemini用に修正)
if api_key_input:
    os.environ["GOOGLE_API_KEY"] = api_key_input
    st.sidebar.success("APIキーが設定されました。")

# --- メイン画面のUI ---
uploaded_file = st.file_uploader("分析したいExcelまたはCSVファイルをアップロードしてください", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # パフォーマンス向上のため、データ読み込みをキャッシュ
        @st.cache_data
        def load_data(file):
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            else:
                return pd.read_excel(file)

        df = load_data(uploaded_file)

        st.write("### アップロードされたデータプレビュー")
        st.dataframe(df.head())

        # ----------------------------------------------------------------------
        # カスタムツールの関数定義
        # df変数が利用できるこの場所でツールを定義する
        # ----------------------------------------------------------------------
        @tool
        def execute_pandas_query(query: str) -> str:
            """
            Executes a Python query on a pandas DataFrame and returns the result as a string.
            The DataFrame 'df' is available in the execution scope.
            The query must assign its final result to a variable named 'result'.
            Example query: "result = df['column_name'].value_counts()"
            """
            try:
                local_vars = {"df": df, "pd": pd}
                exec(query, {"pd": pd}, local_vars)
                result = local_vars.get("result", "Query executed, but no 'result' variable was assigned.")
                return str(result)
            except Exception as e:
                return f"Error executing query: {str(e)}"

        @tool
        def create_and_save_plot(plot_code: str) -> str:
            """
            Executes Python code using Matplotlib/Seaborn to generate a plot and save it as an image file.
            The DataFrame 'df' is available in the execution scope.
            The code must use the 'ax' object for plotting.
            Returns the path to the saved image file.
            Example plot_code: "df['column'].value_counts().plot(kind='bar', ax=ax, title='My Plot')"
            """
            try:
                fig, ax = plt.subplots()
                local_vars = {"df": df, "ax": ax, "plt": plt, "pd": pd}
                exec(plot_code, {"pd": pd}, local_vars)

                # 'static'ディレクトリがなければ作成
                if not os.path.exists("static"):
                    os.makedirs("static")
                
                # ユニークなファイル名を生成して保存
                filename = f"plot_{uuid.uuid4()}.png"
                filepath = os.path.join("static", filename)
                fig.savefig(filepath)
                plt.close(fig)
                return f"Plot successfully generated and saved to '{filepath}'."
            except Exception as e:
                return f"Error generating plot: {str(e)}"

        # ----------------------------------------------------------------------
        # LangChainエージェントの実行部分
        # ----------------------------------------------------------------------
        user_prompt = st.text_area("どのような分析をしますか？ (例: 「東京都の子育て支援に関する投稿を抽出し、センチメントの内訳を教えてください」)", height=150)

        if st.button("分析を実行"):
            if not api_key_input:
                st.error("サイドバーでAPIキーを設定してください。")
            elif not user_prompt:
                st.warning("分析内容を入力してください。")
            else:
                with st.spinner("AIエージェントが分析を実行中です..."):
                    try:
                        # Geminiモデルを初期化 (Gemini用に修正)
                        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0)

                        tools = [execute_pandas_query, create_and_save_plot]
                        
                        # ReActプロンプトテンプレートを取得
                        prompt_template = hub.pull("hwchase17/react")

                        # エージェントを作成
                        agent = create_react_agent(llm, tools, prompt_template)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

                        # DataFrameの情報をプロンプトに含める
                        df_head_str = df.head().to_string()
                        full_prompt = f"""
                        Analyze the following dataframe, whose head is provided below, to answer the user's request.
                        Dataframe Head:
                        {df_head_str}

                        User's Request: {user_prompt}
                        """

                        # エージェントを実行
                        response = agent_executor.invoke({"input": full_prompt})

                        st.write("### 分析結果")
                        st.markdown(response['output'])

                        # レポート内の画像パスを検出し、表示する
                        image_paths = re.findall(r"static/[a-zA-Z0-9_-]+\.png", response['output'])
                        for path in image_paths:
                            if os.path.exists(path):
                                st.image(path)
                        
                        # 思考プロセスを隠せるように修正
                        with st.expander("AIエージェントの思考プロセスを表示"):
                             st.write(response)

                    except Exception as e:
                        st.error(f"エージェントの実行中にエラーが発生しました: {e}")

    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        st.warning("CSVファイルの場合は文字コードがUTF-8であることを確認してください。")