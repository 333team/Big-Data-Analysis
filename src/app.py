import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, ticker
import os
import platform

# ML Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全域設定與工具函式
# ==============================================================================

st.set_page_config(
    page_title="教育大數據分析：24小時線性版",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_chinese_font():
    """獲取中文字體"""
    custom_font_path = "NotoSansTC-Regular.ttf"
    if os.path.exists(custom_font_path):
        return font_manager.FontProperties(fname=custom_font_path)

    system = platform.system()
    if system == "Windows":
        return font_manager.FontProperties(fname=r"C:\Windows\Fonts\msjh.ttc")
    elif system == "Darwin":
        return font_manager.FontProperties(fname="/System/Library/Fonts/PingFang.ttc")
    elif system == "Linux":
        paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        ]
        for p in paths:
            if os.path.exists(p):
                return font_manager.FontProperties(fname=p)
    return None


def set_plot_style():
    """設定繪圖風格"""
    sns.set_style("whitegrid")
    my_font = get_chinese_font()
    if my_font:
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['axes.unicode_minus'] = False
    return my_font


MY_FONT = set_plot_style()


# ==============================================================================
# 1. 資料處理核心邏輯 (鎖定 24 小時)
# ==============================================================================

def load_and_preprocess_data(uploaded_file, remove_outliers=False):
    """
    讀取並清理資料，針對 0~24 小時進行切分
    """
    stats = {}

    try:
        # 0. 讀取資料
        df = pd.read_csv(uploaded_file)
        stats['原始資料'] = len(df)

        # --- 欄位定義 ---
        col_start = '任務派發時間'
        col_submit = '學生首次送出答案的時間點'
        col_score = '首次答題正確率'
        col_duration = '首次答題時間（秒）'
        col_user = '學生姓名去識別化'

        # 1. 時間轉換
        df[col_start] = pd.to_datetime(df[col_start], errors='coerce')
        df[col_submit] = pd.to_datetime(df[col_submit], errors='coerce')

        # 2. 計算 Lag (小時)
        df['lag_hours'] = (df[col_submit] - df[col_start]) / pd.Timedelta(hours=1)

        # 3. 轉數值
        numeric_cols = [
            col_score, col_duration,
            '擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率',
            '文本形式正確率', '文本理解正確率'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4. 基礎清理
        df = df.dropna(subset=['lag_hours', col_score, col_duration])
        stats['移除空值後'] = len(df)

        # 5. 時間範圍過濾：僅保留 0 ~ 24 小時
        df = df[(df['lag_hours'] >= 0) & (df['lag_hours'] <= 24)]
        stats['移除時間異常後'] = len(df)

        # 6. 異常值排除
        if remove_outliers:
            # 分數合理性
            max_score = df[col_score].max()
            upper_limit = 100 if max_score > 1.0 else 1.0
            df = df[(df[col_score] >= 0) & (df[col_score] <= upper_limit)]

            # 時間 IQR
            Q1 = df[col_duration].quantile(0.25)
            Q3 = df[col_duration].quantile(0.75)
            IQR = Q3 - Q1
            time_lower = 1.0
            time_upper = Q3 + 1.5 * IQR

            df = df[(df[col_duration] >= time_lower) & (df[col_duration] <= time_upper)]

            stats['排除極端值後'] = len(df)
        else:
            stats['排除極端值後'] = len(df)

        # 7. 學生分群
        user_stats = df.groupby(col_user)[col_score].mean()
        median_score = user_stats.median()

        def get_group(uid):
            s = user_stats.get(uid)
            if s is None: return '未知'
            return '高分組' if s >= median_score else '潛力組'

        df['ability_group'] = df[col_user].apply(get_group)

        # 8. 自動分箱 (24小時切分)
        # 這裡維持細切，但在線性圖表上，前幾小時的點會擠在一起，這是正常的物理時間呈現
        custom_bins = [
            0, 1, 2, 3, 4, 5, 6,  # 0~6小時
            9, 12, 15, 18, 21, 24  # 6~24小時
        ]

        # 對應的中位數標籤
        bin_labels = [
            0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
            7.5, 10.5, 13.5, 16.5, 19.5, 22.5
        ]

        if len(custom_bins) - 1 != len(bin_labels):
            st.error("分箱錯誤")
            return None, None, None

        df['lag_bin_mid'] = pd.cut(
            df['lag_hours'],
            bins=custom_bins,
            labels=bin_labels,
            include_lowest=True
        )
        df['lag_bin_mid'] = df['lag_bin_mid'].astype(float)

        return df, median_score, stats

    except Exception as e:
        st.error(f"資料處理發生錯誤: {e}")
        return None, None, None


# ==============================================================================
# 2. 主程式介面
# ==============================================================================

def main():
    st.sidebar.title("設定面板")

    col_user = '學生姓名去識別化'
    col_score = '首次答題正確率'
    col_duration = '首次答題時間（秒）'
    col_difficulty = '難易度'
    col_task = '任務名稱'

    st.markdown("## 📊 教育數據分析：24小時黃金窗口 (線性時間軸)")
    st.info("本系統鎖定 **0 ~ 24 小時** 的數據，X 軸採用均勻時間顯示。")

    uploaded_file = st.sidebar.file_uploader("📂 上傳 CSV 資料檔", type="csv")

    st.sidebar.markdown("---")
    st.sidebar.subheader("資料清理")
    enable_outlier_removal = st.sidebar.checkbox("排除異常值 (Outlier Removal)", value=True)

    if uploaded_file is None:
        st.warning("👈 請先上傳資料。")
        return

    with st.spinner("正在分析 24 小時內的數據..."):
        df, median_score, stats = load_and_preprocess_data(uploaded_file, remove_outliers=enable_outlier_removal)

    if df is None: return

    total_removed = stats['原始資料'] - stats['排除極端值後']
    st.success(f"✅ 分析完成！有效樣本：{len(df)} 筆 (已移除 24 小時以外及異常資料共 {total_removed} 筆)")

    with st.expander("查看詳細清理報告"):
        st.write(stats)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 資料概覽",
        "📉 24H 鞏固曲線",
        "⏱️ 24H 認知負荷",
        "👥 分群/題型差異",
        "🤖 AI 預測模型"
    ])

    # --- Tab 1 ---
    with tab1:
        st.subheader("24小時內資料概覽")
        st.dataframe(df.head(10))
        col1, col2, col3 = st.columns(3)
        col1.metric("分析區間", "0 - 24 小時")
        col2.metric("學生總數", df[col_user].nunique())
        col3.metric("平均答題時間", f"{df[col_duration].mean():.1f} 秒")

    # --- Tab 2: 24H 鞏固曲線 (線性版) ---
    with tab2:
        st.subheader("驗證：24小時內的記憶鞏固趨勢 (線性時間軸)")

        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            y_axis_option = st.selectbox(
                "選擇分析指標 (Y軸)",
                [col_score, '擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率', '文本理解正確率']
            )
        with col_ctrl2:
            split_by_diff = st.checkbox("依「難易度」分層", value=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        if split_by_diff and col_difficulty in df.columns:
            diff_order = ['易', '中', '難']
            colors = {'易': '#2ecc71', '中': '#f39c12', '難': '#e74c3c'}

            unique = df[col_difficulty].unique()
            sorted_diffs = [d for d in diff_order if d in unique] + [d for d in unique if d not in diff_order]

            for diff in sorted_diffs:
                sub = df[df[col_difficulty] == diff]
                if len(sub) == 0: continue
                agg = sub.groupby('lag_bin_mid')[y_axis_option].mean().reset_index()
                ax.plot(agg['lag_bin_mid'], agg[y_axis_option], color=colors.get(diff, 'gray'), lw=2, marker='o',
                        label=f'{diff} ({len(sub)}筆)')
        else:
            agg_data = df.groupby('lag_bin_mid')[y_axis_option].agg(['mean', 'count']).reset_index()
            ax.plot(agg_data['lag_bin_mid'], agg_data['mean'], color='royalblue', lw=2, marker='o')
            for x, y, c in zip(agg_data['lag_bin_mid'], agg_data['mean'], agg_data['count']):
                ax.text(x, y + 0.005, f"{y:.2f}\n(n={c})", fontsize=8, ha='center', va='bottom')

        # --- 【修改點】X 軸設定為線性 ---
        # 移除 ax.set_xscale('log')
        # 設定均勻刻度：每 3 小時一格
        linear_ticks = np.arange(0, 25, 3)  # [0, 3, 6, 9, 12, 15, 18, 21, 24]
        ax.set_xticks(linear_ticks)

        # Y 軸縮放 (若希望波動明顯，可改為 (0.3, 0.7))
        ax.set_ylim(0, 1.1)

        ax.set_title(f"24小時記憶鞏固趨勢：{y_axis_option}", fontproperties=MY_FONT, fontsize=16)
        ax.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
        ax.set_ylabel("平均正確率", fontproperties=MY_FONT)
        ax.legend(prop=MY_FONT)
        ax.grid(True, which="both", alpha=0.3)

        st.pyplot(fig)

    # --- Tab 3: 24H 認知負荷 (線性版) ---
    with tab3:
        st.subheader("驗證：24小時內的反應時間變化")
        split_time = st.checkbox("依「難易度」分層檢視", value=True, key='time')

        fig2, ax2 = plt.subplots(figsize=(12, 6))

        if split_time and col_difficulty in df.columns:
            colors = {'易': '#2ecc71', '中': '#f39c12', '難': '#e74c3c'}
            unique = df[col_difficulty].unique()
            sorted_diffs = [d for d in ['易', '中', '難'] if d in unique]

            for diff in sorted_diffs:
                sub = df[df[col_difficulty] == diff]
                if len(sub) == 0: continue
                agg = sub.groupby('lag_bin_mid')[col_duration].median().reset_index()
                ax2.plot(agg['lag_bin_mid'], agg[col_duration], color=colors.get(diff, 'gray'), marker='s', label=diff)
        else:
            agg = df.groupby('lag_bin_mid')[col_duration].median().reset_index()
            ax2.plot(agg['lag_bin_mid'], agg[col_duration], color='orange', marker='s')

        # --- 【修改點】X 軸設定為線性 ---
        ax2.set_xticks(np.arange(0, 25, 3))

        ax2.set_title("24小時認知負荷 (反應時間)", fontproperties=MY_FONT, fontsize=16)
        ax2.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
        ax2.legend(prop=MY_FONT)
        ax2.grid(True, which="both", alpha=0.3)
        st.pyplot(fig2)

    # --- Tab 4: 分群差異 (線性版) ---
    with tab4:
        st.subheader("分群差異 (0-24H)")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### 高分組 vs 潛力組")
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            for group in ['高分組', '潛力組']:
                sub = df[df['ability_group'] == group]
                agg = sub.groupby('lag_bin_mid')[col_score].mean().reset_index()
                ax3.plot(agg['lag_bin_mid'], agg[col_score], marker='o', label=group)

            # 線性軸
            ax3.set_xticks(np.arange(0, 25, 6))  # 每6小時一格
            ax3.set_ylim(0, 1.1)
            ax3.legend(prop=MY_FONT)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)

        with col_d2:
            st.markdown("##### 知識向度")
            know_cols = st.multiselect("選擇向度", ['擷取訊息正確率', '發展解釋正確率'], default=['擷取訊息正確率'])
            if know_cols:
                fig4, ax4 = plt.subplots(figsize=(6, 5))
                for col in know_cols:
                    if col in df.columns:
                        agg = df.groupby('lag_bin_mid')[col].mean().reset_index()
                        ax4.plot(agg['lag_bin_mid'], agg[col], marker='.', label=col.replace('正確率', ''))
                # 線性軸
                ax4.set_xticks(np.arange(0, 25, 6))
                ax4.set_ylim(0, 1.1)
                ax4.legend(prop=MY_FONT)
                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)

    # --- Tab 5: AI 預測 ---
    with tab5:
        st.subheader("AI 預測模型 (24H 短期預測)")
        if st.button("訓練模型"):
            with st.spinner("Training..."):
                model_df = df.copy()
                le = LabelEncoder()
                model_df['diff_code'] = le.fit_transform(model_df[col_difficulty].astype(str))
                model_df['log_lag'] = np.log1p(model_df['lag_hours'])
                model_df['user_ability'] = model_df.groupby(col_user)[col_score].transform('mean')

                score_max = model_df[col_score].max()
                thresh = 80 if score_max > 1.0 else 0.8
                model_df['target'] = np.where(model_df[col_score] < thresh, 1, 0)

                features = ['lag_hours', 'diff_code', 'user_ability', col_duration]
                if '年級' in model_df.columns:
                    model_df['grade_code'] = le.fit_transform(model_df['年級'].astype(str))
                    features.append('grade_code')

                model_df = model_df.dropna(subset=features)
                clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
                clf.fit(model_df[features], model_df['target'])

                st.success("模型訓練完成")

                imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                imp.plot(kind='barh', ax=ax_imp, color='teal')
                ax_imp.set_title("24H內影響因子", fontproperties=MY_FONT)
                st.pyplot(fig_imp)


if __name__ == "__main__":
    main()