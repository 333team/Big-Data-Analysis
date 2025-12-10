import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import os
import platform

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 0. 全域設定與工具函式
# ==============================================================================

st.set_page_config(
    page_title="教育大數據分析平台：實務整合版",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_chinese_font():
    """
    嘗試獲取中文字體，優先尋找專案目錄下的字體，若無則嘗試系統字體
    """
    # 1. 優先路徑
    custom_font_path = "NotoSansTC-Regular.ttf"  # 假設您有放這個檔案，沒有也沒關係
    if os.path.exists(custom_font_path):
        return font_manager.FontProperties(fname=custom_font_path)

    # 2. 系統字體嘗試
    system = platform.system()
    if system == "Windows":
        return font_manager.FontProperties(fname=r"C:\Windows\Fonts\msjh.ttc")  # 微軟正黑體
    elif system == "Darwin":  # Mac
        return font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")  # 蘋方
    elif system == "Linux":
        # Linux 常見中文字體路徑，可視情況新增
        paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        ]
        for p in paths:
            if os.path.exists(p):
                return font_manager.FontProperties(fname=p)

    return None


def set_plot_style():
    """設定繪圖風格與字體"""
    sns.set_style("whitegrid")
    my_font = get_chinese_font()

    if my_font:
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['axes.unicode_minus'] = False

    return my_font


# 初始化字體
MY_FONT = set_plot_style()


# ==============================================================================
# 1. 資料處理核心邏輯
# ==============================================================================

def load_and_preprocess_data(uploaded_file):
    """讀取並清理資料"""
    try:
        df = pd.read_csv(uploaded_file)

        # --- 欄位名稱對照 (依據您的需求) ---
        # 為了程式方便操作，將中文欄位映射到英文變數名，但保留原欄位
        # 關鍵時間欄位
        col_start = '任務派發時間'
        col_submit = '學生首次送出答案的時間點'
        col_score = '首次答題正確率'
        col_duration = '首次答題時間（秒）'
        col_user = '學生姓名去識別化'
        col_difficulty = '難易度'

        # 1. 時間轉換
        df[col_start] = pd.to_datetime(df[col_start], errors='coerce')
        df[col_submit] = pd.to_datetime(df[col_submit], errors='coerce')

        # 2. 計算 Lag (小時)
        df['lag_hours'] = (df[col_submit] - df[col_start]) / pd.Timedelta(hours=1)

        # 3. 數值清理
        numeric_cols = [
            '首次答題正確率', '首次答題時間（秒）',
            '擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率',
            '文本形式正確率', '文本理解正確率'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 移除無效資料 (時間無法計算或正確率遺失)
        df = df.dropna(subset=['lag_hours', col_score])

        # 5. 過濾極端值 (例如只看 0 ~ 7天內的資料，避免長尾效應影響圖表)
        df = df[(df['lag_hours'] >= 0) & (df['lag_hours'] <= 168)]

        # 6. 學生能力分群 (基於 '首次答題正確率' 的中位數)
        user_stats = df.groupby(col_user)[col_score].mean()
        median_score = user_stats.median()

        def get_group(uid):
            s = user_stats.get(uid)
            if s is None: return '未知'
            return '高分組' if s >= median_score else '潛力組'  # 改名為潛力組比較好聽

        df['ability_group'] = df[col_user].apply(get_group)

        # 7. 自動分箱 (為了畫折線圖) - 每 4 小時一格
        # 建立一個 binned column 用於繪圖聚合
        df['lag_bin_mid'] = (df['lag_hours'] // 4) * 4 + 2  # 取中間值

        return df, median_score

    except Exception as e:
        st.error(f"資料處理發生錯誤: {e}")
        return None, None


# ==============================================================================
# 2. 主程式介面
# ==============================================================================

def main():
    st.sidebar.title("設定面板")

    # --- 【修正點】在此定義欄位變數，讓整個 main 函式都能使用 ---
    col_user = '學生姓名去識別化'
    col_score = '首次答題正確率'
    col_duration = '首次答題時間（秒）'
    col_difficulty = '難易度'
    col_task = '任務名稱'
    # -------------------------------------------------------

    st.markdown("## 📊 教育數據分析：記憶鞏固與適度困難")
    st.info("本系統將理論模擬功能與實務分析合併，使用真實數據驗證學習理論。")

    # 1. 檔案上傳
    uploaded_file = st.sidebar.file_uploader("📂 上傳 CSV 資料檔", type="csv")

    if uploaded_file is None:
        st.warning("👈 請先在左側上傳資料以開始分析。")
        st.markdown("#### 預期包含欄位：")
        st.code(
            "地區, 年級, 學生姓名去識別化, 難易度, 任務派發時間, 學生首次送出答案的時間點, 首次答題時間（秒）, 首次答題正確率, 擷取訊息正確率...",
            language="csv")
        return

    # 2. 資料載入
    with st.spinner("正在讀取並分析資料..."):
        # 這裡會呼叫上面的函式，回傳處理好的 df
        df, median_score = load_and_preprocess_data(uploaded_file)

    if df is None:
        return

    st.success(f"✅ 資料載入成功！分析樣本數：{len(df)} 筆 (能力分群中位數: {median_score:.2f})")

    # 建立頁籤
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 資料概覽",
        "📉 記憶鞏固曲線 (U-Curve)",
        "⏱️ 認知負荷 (反應時間)",
        "👥 分群/題型差異",
        "🤖 AI 預測模型"
    ])

    # --- Tab 1: 資料概覽 ---
    with tab1:
        st.subheader("原始資料預覽")
        st.dataframe(df.head(10))

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("學生總數", df[col_user].nunique())  # 使用變數
        col_m2.metric("任務總數", df[col_task].nunique())  # 使用變數
        col_m3.metric("平均答題時間", f"{df[col_duration].mean():.1f} 秒")  # 使用變數

    # --- Tab 2: 記憶鞏固曲線 ---
    with tab2:
        st.subheader("驗證：練習延遲時間 vs. 正確率")
        # ... (中間程式碼不變) ...
        # 這裡原本就沒有用到 col_user，所以不影響

        # 選擇 Y 軸指標
        y_axis_option = st.selectbox(
            "選擇分析指標 (Y軸)",
            [col_score, '擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率', '文本理解正確率'],
            index=0
        )

        agg_data = df.groupby('lag_bin_mid')[y_axis_option].mean().reset_index()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(agg_data['lag_bin_mid'], agg_data[y_axis_option], color='gray', alpha=0.5)
        ax.plot(agg_data['lag_bin_mid'], agg_data[y_axis_option], color='royalblue', lw=2, marker='o')

        ax.set_title(f"記憶鞏固趨勢：{y_axis_option}", fontproperties=MY_FONT, fontsize=16)
        ax.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
        ax.set_ylabel(f"平均{y_axis_option}", fontproperties=MY_FONT)
        st.pyplot(fig)
        st.info("💡 **觀察重點**：曲線是否在某個時間點（如 6~24 小時）出現高峰？")

    # --- Tab 3: 認知負荷分析 ---
    with tab3:
        st.subheader("驗證：練習延遲時間 vs. 反應時間")
        agg_time = df.groupby('lag_bin_mid')[col_duration].median().reset_index()  # 使用變數

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(agg_time['lag_bin_mid'], agg_time[col_duration], color='orange', lw=2, marker='s')

        ax2.set_title("時間延遲對認知負荷的影響", fontproperties=MY_FONT, fontsize=16)
        ax2.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
        ax2.set_ylabel("首次答題時間中位數 (秒)", fontproperties=MY_FONT)
        st.pyplot(fig2)

    # --- Tab 4: 分群與知識類型差異 ---
    with tab4:
        st.subheader("深入分析：不同族群與題型")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### 1. 高分組 vs 潛力組")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            for group in ['高分組', '潛力組']:
                sub_df = df[df['ability_group'] == group]
                agg = sub_df.groupby('lag_bin_mid')[col_score].mean().reset_index()  # 使用變數
                agg = agg[agg['lag_bin_mid'] <= 72]
                ax3.plot(agg['lag_bin_mid'], agg[col_score], marker='o', label=group)

            ax3.set_title("不同能力組的鞏固曲線", fontproperties=MY_FONT)
            ax3.set_xlabel("延遲 (小時)", fontproperties=MY_FONT)
            ax3.legend(prop=MY_FONT)
            st.pyplot(fig3)

        with col_d2:
            st.markdown("##### 2. 知識向度比較")
            know_cols = st.multiselect(
                "選擇要比較的知識向度",
                ['擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率', '文本理解正確率'],
                default=['擷取訊息正確率', '發展解釋正確率']
            )
            if know_cols:
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                for col in know_cols:
                    if col in df.columns:
                        agg = df.groupby('lag_bin_mid')[col].mean().reset_index()
                        agg = agg[agg['lag_bin_mid'] <= 72]
                        ax4.plot(agg['lag_bin_mid'], agg[col], marker='.', label=col.replace('正確率', ''))
                ax4.set_title("不同知識類型的遺忘/鞏固趨勢", fontproperties=MY_FONT)
                ax4.legend(prop=MY_FONT)
                st.pyplot(fig4)

    # --- Tab 5: 機器學習預測 (這是原本報錯的地方) ---
    with tab5:
        st.subheader("隨機森林預測模型")
        st.markdown("利用真實數據訓練模型，預測學生是否需要幫助。")

        if st.button("🚀 開始訓練模型"):
            with st.spinner("模型訓練中..."):
                model_df = df.copy()

                le = LabelEncoder()
                model_df['diff_code'] = le.fit_transform(model_df[col_difficulty].astype(str))

                model_df['log_lag'] = np.log1p(model_df['lag_hours'])

                # 【修正點】這裡原本因為找不到 col_user 和 col_score 而報錯
                # 現在因為我們在 main 開頭定義了，所以這裡可以正常運作
                model_df['user_ability'] = model_df.groupby(col_user)[col_score].transform('mean')

                threshold = 80 if model_df[col_score].max() > 1 else 0.8
                model_df['target'] = np.where(model_df[col_score] < threshold, 1, 0)

                features = ['lag_hours', 'diff_code', 'user_ability', col_duration]
                if '年級' in model_df.columns:
                    model_df['grade_code'] = le.fit_transform(model_df['年級'].astype(str))
                    features.append('grade_code')

                model_df = model_df.dropna(subset=features)

                X = model_df[features]
                y = model_df['target']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
                clf.fit(X_train, y_train)

                y_pred = clf.predict(X_test)
                acc = clf.score(X_test, y_test)

                st.success(f"模型訓練完成！ 準確率: {acc:.2%}")

                # 特徵重要性繪圖
                st.markdown("#### 關鍵影響因子 (Feature Importance)")
                importances = clf.feature_importances_
                indices = np.argsort(importances)[::-1]
                names = [features[i] for i in indices]

                fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                sns.barplot(x=importances[indices], y=names, palette='viridis', ax=ax_imp)
                ax_imp.set_title("哪些因素最影響答題結果？", fontproperties=MY_FONT)
                st.pyplot(fig_imp)

                # 混淆矩陣
                st.markdown("#### 預測結果混淆矩陣")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=['預測:通過', '預測:需補救'],
                            yticklabels=['實際:通過', '實際:需補救'], ax=ax_cm)
                ax_cm.set_xlabel('模型預測', fontproperties=MY_FONT)
                ax_cm.set_ylabel('真實情況', fontproperties=MY_FONT)
                plt.xticks(fontproperties=MY_FONT)
                plt.yticks(fontproperties=MY_FONT)
                st.pyplot(fig_cm)


if __name__ == "__main__":
    main()