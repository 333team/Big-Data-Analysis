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
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 0. 全域設定與 CSS 美化 (深色模式修復版)
# ==============================================================================

st.set_page_config(
    page_title="教育大數據｜學習歷程分析儀表板",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 專業級 CSS 樣式 (支援深色/淺色模式) ---
st.markdown("""
    <style>
    /* 全局字體 */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
    }

    /* 主標題樣式 - 使用漸層文字 */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        /* 漸層在深色模式下可能對比度不足，增加文字陰影輔助 */
        background: linear-gradient(90deg, #4F8BF9 0%, #8E44AD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }

    .sub-title {
        font-size: 1rem;
        /* 使用 Streamlit 變數，讓它在深色模式自動變白 */
        color: var(--text-color);
        opacity: 0.8;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* KPI 卡片樣式 - 自動適應深色模式 */
    .kpi-card {
        /* 使用 secondary-background-color，在淺色是淺灰/白，深色是深灰 */
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        /* 邊框顏色保持 */
        border-top-width: 4px;
        border-top-style: solid;
        margin-bottom: 10px;
    }

    .kpi-title { 
        font-size: 0.85rem; 
        color: var(--text-color); 
        opacity: 0.7;
        text-transform: uppercase; 
        margin-bottom: 5px; 
    }

    .kpi-value { 
        font-size: 1.8rem; 
        font-weight: 700; 
        color: var(--text-color); 
    }

    .kpi-note { 
        font-size: 0.8rem; 
        color: #27AE60; /* 綠色在深淺色都看得到 */
        font-weight: 500;
    }

    /* Tabs 美化 */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid rgba(128,128,128,0.2); }
    .stTabs [data-baseweb="tab"] {
        height: 45px; 
        background-color: transparent; 
        border-radius: 6px 6px 0px 0px;
        padding: 0 20px; 
        font-weight: 500; 
        color: var(--text-color); /* 自動適應文字顏色 */
        border: none;
        opacity: 0.7;
    }
    .stTabs [aria-selected="true"] {
        /* 選中時的背景，使用次要背景色 */
        background-color: var(--secondary-background-color); 
        color: var(--primary-color); 
        border-bottom: 3px solid var(--primary-color);
        opacity: 1;
    }

    /* 資訊區塊樣式 */
    .info-box {
        background-color: var(--secondary-background-color);
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid rgba(128,128,128,0.2);
    }

    /* 預測結果卡片 */
    .prediction-result {
        padding: 20px; border-radius: 12px; color: white; text-align: center; margin-top: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    /* 保持原有的漸層，因為文字是白色，所以在任何模式下都看得到 */
    .pred-danger { background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%); }
    .pred-safe { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }

    </style>
    """, unsafe_allow_html=True)


def get_chinese_font():
    """獲取中文字體：優先使用專案目錄下的字型檔"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_name = "NotoSansTC-Regular.ttf"
    font_path = os.path.join(current_dir, font_name)
    if os.path.exists(font_path):
        return font_manager.FontProperties(fname=font_path)

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
    """設定全域繪圖風格"""
    # 使用 whitegrid，但在繪圖時我們會強制加上白色背景，確保深色模式下圖表依然清晰
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    my_font = get_chinese_font()

    if my_font:
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set(font=my_font.get_name())
        return my_font
    else:
        plt.rcParams['axes.unicode_minus'] = False
        return None


MY_FONT = set_plot_style()


def display_kpi_card(title, value, note, color_border="#4F8BF9"):
    # 這裡的 CSS class 已經改為使用變數，所以 Python 這裡不需要動
    st.markdown(f"""
    <div class="kpi-card" style="border-top-color: {color_border};">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 1. 資料處理核心邏輯 (0-24H 高密度版)
# ==============================================================================

@st.cache_data
def load_and_preprocess_data(uploaded_file, remove_outliers=False):
    stats = {}
    try:
        df = pd.read_csv(uploaded_file)
        stats['original_count'] = len(df)

        col_start = '任務派發時間'
        col_submit = '學生首次送出答案的時間點'
        col_score = '首次答題正確率'
        col_duration = '首次答題時間（秒）'
        col_user = '學生姓名去識別化'

        required_cols = [col_start, col_submit, col_score, col_duration, col_user]
        if not all(col in df.columns for col in required_cols):
            st.error(f"資料缺少必要欄位：{required_cols}")
            return None, None, None

        df[col_start] = pd.to_datetime(df[col_start], errors='coerce')
        df[col_submit] = pd.to_datetime(df[col_submit], errors='coerce')
        df['lag_hours'] = (df[col_submit] - df[col_start]) / pd.Timedelta(hours=1)

        numeric_cols = [col_score, col_duration, '擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率',
                        '文本理解正確率']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['lag_hours', col_score, col_duration])
        df = df[(df['lag_hours'] >= 0) & (df['lag_hours'] <= 24)]
        stats['valid_24h_count'] = len(df)

        if remove_outliers:
            max_score = df[col_score].max()
            upper_limit = 100 if max_score > 1.0 else 1.0
            df = df[(df[col_score] >= 0) & (df[col_score] <= upper_limit)]

            Q1 = df[col_duration].quantile(0.25)
            Q3 = df[col_duration].quantile(0.75)
            IQR = Q3 - Q1
            time_upper = Q3 + 1.5 * IQR
            df = df[(df[col_duration] >= 1.0) & (df[col_duration] <= time_upper)]

        stats['final_count'] = len(df)

        if len(df) == 0:
            st.error("篩選後無有效資料，請檢查資料格式或調整篩選條件。")
            return None, None, None

        user_stats = df.groupby(col_user)[col_score].mean()
        median_score = user_stats.median()

        def get_group(uid):
            return '高分組' if user_stats.get(uid, 0) >= median_score else '潛力組'

        df['ability_group'] = df[col_user].apply(get_group)

        custom_bins = [0, 1, 2, 3, 4, 5, 6, 9, 12, 15, 18, 21, 24]
        bin_labels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 10.5, 13.5, 16.5, 19.5, 22.5]

        df['lag_bin_mid'] = pd.cut(df['lag_hours'], bins=custom_bins, labels=bin_labels, include_lowest=True)
        df['lag_bin_mid'] = df['lag_bin_mid'].astype(float)

        return df, median_score, stats

    except Exception as e:
        st.error(f"資料處理錯誤: {e}")
        return None, None, None


# ==============================================================================
# 2. 主程式介面
# ==============================================================================

def main():
    if 'trained_model' not in st.session_state:
        st.session_state['trained_model'] = None
    if 'label_encoder' not in st.session_state:
        st.session_state['label_encoder'] = None
    if 'model_features' not in st.session_state:
        st.session_state['model_features'] = None

    # --- Sidebar ---
    with st.sidebar:
        st.title("控制台")
        uploaded_file = st.file_uploader("📂 上傳 CSV 資料", type="csv")
        st.markdown("### ⚙️ 參數設定")
        enable_outlier_removal = st.toggle("IQR 極端值過濾", value=True)
        st.info("ℹ️ 分析範圍鎖定：0 ~ 24 小時")
        st.caption("Auto Dark/Light Mode Supported")

    # --- Header ---
    st.markdown('<div class="main-title">🎓 教育大數據：學習黃金窗口</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">24H Learning Consolidation Analytics Dashboard</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.info("👋 請先從左側面板上傳學習歷程資料以開始分析。")
        return

    with st.spinner("🚀 正在進行數據清洗與特徵工程..."):
        df, median_score, stats = load_and_preprocess_data(uploaded_file, enable_outlier_removal)

    if df is None: return

    col_score = '首次答題正確率'
    col_duration = '首次答題時間（秒）'
    col_difficulty = '難易度'
    col_user = '學生姓名去識別化'

    avg_score = df[col_score].mean()
    if avg_score <= 1.0:
        score_fmt = f"{avg_score * 100:.1f}%"
    else:
        score_fmt = f"{avg_score:.1f}"

    # --- KPI Section (CSS will handle colors) ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("有效樣本 (24H)", f"{len(df):,}", f"資料保留率: {len(df) / stats['original_count']:.1%}",
                         "#3498db")
    with col2:
        display_kpi_card("不重複學生", f"{df[col_user].nunique():,}", "Active Learners", "#9b59b6")
    with col3:
        display_kpi_card("平均正確率", score_fmt, "Overall Accuracy", "#2ecc71")
    with col4:
        display_kpi_card("平均答題耗時", f"{df[col_duration].mean():.1f}s", "Avg Duration", "#f1c40f")

    st.markdown("---")

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 資料概覽", "📉 鞏固曲線", "⏱️ 認知負荷", "👥 群體差異", "🤖 AI 預測"
    ])

    # Tab 1: 資料概覽
    with tab1:
        st.subheader("🔍 數據健康度檢測")
        col_stat1, col_stat2 = st.columns([3, 1])
        with col_stat1:
            st.dataframe(df.head(100), use_container_width=True, height=400)
        with col_stat2:
            st.markdown(f"""
                <div class="info-box">
                    <h4 style="margin-top:0;">🧹 清理摘要</h4>
                    <p><b>原始筆數：</b> {stats['original_count']:,}</p>
                    <p><b>24H樣本：</b> {stats['valid_24h_count']:,}</p>
                    <p><b>最終分析：</b> {stats['final_count']:,}</p>
                    <hr style="border-top: 1px solid rgba(128,128,128,0.2);">
                    <small style="opacity:0.8;">排除條件：時間異常、數值錯誤、IQR極端值。</small>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: 鞏固曲線
    with tab2:
        st.subheader("📉 記憶鞏固趨勢分析")
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            y_opt = st.selectbox("分析指標 (Y軸)", [col_score, '擷取訊息正確率', '發展解釋正確率'])
            split_diff = st.toggle("依難易度分層", value=True)

        with col_ctrl2:
            fig, ax = plt.subplots(figsize=(10, 5))

            # 強制設定白色背景，確保在深色模式下看得見黑色文字與格線
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            if split_diff and col_difficulty in df.columns:
                diff_order = ['易', '中', '難']
                colors = {'易': '#27ae60', '中': '#f39c12', '難': '#c0392b'}
                present_diffs = [d for d in diff_order if d in df[col_difficulty].unique()]

                agg = df.groupby(['lag_bin_mid', col_difficulty])[y_opt].mean().reset_index()
                agg = agg[agg[col_difficulty].isin(present_diffs)]

                sns.lineplot(data=agg, x='lag_bin_mid', y=y_opt, hue=col_difficulty,
                             hue_order=present_diffs, palette=colors,
                             marker='o', linewidth=2.5, ax=ax)
            else:
                agg = df.groupby('lag_bin_mid')[y_opt].mean().reset_index()
                sns.lineplot(data=agg, x='lag_bin_mid', y=y_opt, marker='o',
                             color='#2980b9', linewidth=3, label="全體平均", ax=ax)

            ax.set_xticks(np.arange(0, 25, 3))
            ax.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
            ax.set_ylabel("平均分數", fontproperties=MY_FONT)
            ax.set_title(f"24小時內 {y_opt} 變化趨勢", fontproperties=MY_FONT, fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.5)
            if MY_FONT: ax.legend(prop=MY_FONT)
            st.pyplot(fig)

        with st.expander("💡 圖表解讀"):
            st.markdown("""
            *   **趨勢意義**：觀察曲線是否隨時間上升。若上升，代表存在「記憶鞏固」效應。
            """)

    # Tab 3: 認知負荷
    with tab3:
        st.subheader("⏱️ 認知負荷 (答題時間) 分析")

        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            split_time_diff = st.toggle("依難易度分層", value=True, key="time_split")

        with col_t2:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            fig2.patch.set_facecolor('white')
            ax2.set_facecolor('white')

            if split_time_diff and col_difficulty in df.columns:
                diff_order = ['易', '中', '難']
                colors = {'易': '#27ae60', '中': '#f39c12', '難': '#c0392b'}
                present_diffs = [d for d in diff_order if d in df[col_difficulty].unique()]

                agg = df.groupby(['lag_bin_mid', col_difficulty])[col_duration].median().reset_index()
                agg = agg[agg[col_difficulty].isin(present_diffs)]

                sns.lineplot(data=agg, x='lag_bin_mid', y=col_duration, hue=col_difficulty,
                             hue_order=present_diffs, palette=colors,
                             marker='s', linewidth=2.5, ax=ax2)
            else:
                agg = df.groupby('lag_bin_mid')[col_duration].median().reset_index()
                ax2.fill_between(agg['lag_bin_mid'], agg[col_duration], color="#f39c12", alpha=0.1)
                sns.lineplot(data=agg, x='lag_bin_mid', y=col_duration, marker='s',
                             color='#e67e22', linewidth=2.5, label="全體中位數", ax=ax2)

            ax2.set_xticks(np.arange(0, 25, 3))
            ax2.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
            ax2.set_ylabel("答題時間中位數 (秒)", fontproperties=MY_FONT)
            ax2.grid(True, linestyle='--', alpha=0.5)
            if MY_FONT: ax2.legend(prop=MY_FONT)
            st.pyplot(fig2)

    # Tab 4: 分群差異
    with tab4:
        st.subheader("👥 學習者分群行為差異")
        col_d1, col_d2 = st.columns(2)

        # 左邊：高分 vs 潛力
        with col_d1:
            st.markdown("##### 1. 高分組 vs 潛力組 (正確率)")
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            fig3.patch.set_facecolor('white')
            ax3.set_facecolor('white')

            for g, c in zip(['高分組', '潛力組'], ['#2980b9', '#e74c3c']):
                sub = df[df['ability_group'] == g]
                agg = sub.groupby('lag_bin_mid')[col_score].mean().reset_index()
                sns.lineplot(data=agg, x='lag_bin_mid', y=col_score, marker='o', label=g, color=c, linewidth=2, ax=ax3)

            ax3.set_xticks(np.arange(0, 25, 6))
            ax3.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)

            # --- 【修正點 1】強制設定 Y 軸標籤字體 ---
            ax3.set_ylabel("平均正確率", fontproperties=MY_FONT)

            # --- 【修正點 2】強制設定圖例字體 ---
            if MY_FONT:
                ax3.legend(prop=MY_FONT)

            st.pyplot(fig3)

        # 右邊：知識向度
        with col_d2:
            st.markdown("##### 2. 知識向度差異")
            candidate_cols = ['擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率', '文本理解正確率']
            valid_options = [c for c in candidate_cols if c in df.columns]

            know_cols = st.multiselect(
                "請選擇要顯示的向度 (可多選):",
                options=valid_options,
                default=[valid_options[0]] if valid_options else None
            )

            fig4, ax4 = plt.subplots(figsize=(6, 5))
            fig4.patch.set_facecolor('white')
            ax4.set_facecolor('white')

            if know_cols:
                markers = ['o', 's', '^', 'D', 'v']
                for idx, col in enumerate(know_cols):
                    agg = df.groupby('lag_bin_mid')[col].mean().reset_index()
                    label_name = col.replace('正確率', '')
                    marker = markers[idx % len(markers)]
                    sns.lineplot(data=agg, x='lag_bin_mid', y=col, marker=marker, label=label_name, linewidth=2, ax=ax4)

                ax4.set_xticks(np.arange(0, 25, 6))
                ax4.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)

                # --- 【修正點 3】強制設定 Y 軸標籤字體 ---
                ax4.set_ylabel("平均正確率", fontproperties=MY_FONT)

                # --- 【修正點 4】強制設定圖例字體 ---
                if MY_FONT:
                    ax4.legend(prop=MY_FONT)

                ax4.grid(True, alpha=0.3)
                st.pyplot(fig4)
            else:
                st.info("請從上方選單至少選擇一個向度以顯示圖表。")

    # Tab 5: AI 預測
    with tab5:
        st.subheader("🤖 AI 學習風險預測模型")
        st.caption("基於隨機森林 (Random Forest) 演算法，預測學生在特定情境下的學習風險。")

        with st.container():
            col_train_btn, col_train_info = st.columns([1, 4])
            with col_train_btn:
                train_btn = st.button("🚀 訓練模型", type="primary", use_container_width=True)

            if train_btn:
                with st.spinner("正在訓練模型並計算特徵權重..."):
                    model_df = df.copy()
                    le = LabelEncoder()

                    if col_difficulty in model_df.columns:
                        model_df['diff_code'] = le.fit_transform(model_df[col_difficulty].astype(str))
                    else:
                        model_df['diff_code'] = 0

                    model_df['user_ability'] = model_df.groupby(col_user)[col_score].transform('mean')
                    thresh = 80 if model_df[col_score].max() > 1.0 else 0.8
                    model_df['target'] = np.where(model_df[col_score] < thresh, 1, 0)

                    feats = ['lag_hours', 'diff_code', 'user_ability', col_duration]
                    model_df = model_df.dropna(subset=feats)

                    X = model_df[feats]
                    y = model_df['target']

                    if len(X) > 50:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        clf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced')
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)

                        st.session_state['trained_model'] = clf
                        st.session_state['label_encoder'] = le
                        st.session_state['model_features'] = feats
                        st.session_state['accuracy'] = accuracy_score(y_test, y_pred)
                        st.session_state['y_test'] = y_test
                        st.session_state['y_pred'] = y_pred

                        st.toast(f"模型訓練完成！準確率: {st.session_state['accuracy']:.2%}", icon="✅")
                    else:
                        st.error("樣本不足，無法訓練模型。")

        if st.session_state['trained_model'] is not None:
            col_plot1, col_plot2 = st.columns(2)
            clf = st.session_state['trained_model']
            feats = st.session_state['model_features']

            with col_plot1:
                st.markdown("##### 🔑 影響因子權重")
                name_mapping = {
                    'lag_hours': '練習延遲時間',
                    'diff_code': '任務難易度',
                    'user_ability': '學生程度',
                    col_duration: '答題耗時'
                }
                imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
                imp.index = [name_mapping.get(x, x) for x in imp.index]

                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                fig_imp.patch.set_facecolor('white')
                ax_imp.set_facecolor('white')

                imp.plot(kind='barh', ax=ax_imp, color='#16a085', width=0.7)
                if MY_FONT:
                    ax_imp.set_yticklabels(imp.index, fontproperties=MY_FONT, fontsize=11)
                    ax_imp.set_xlabel("Importance", fontproperties=MY_FONT)
                st.pyplot(fig_imp)

            with col_plot2:
                st.markdown("##### 🔍 預測混淆矩陣")
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                fig_cm.patch.set_facecolor('white')
                ax_cm.set_facecolor('white')

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                            cbar=False, annot_kws={"size": 14},
                            xticklabels=['通過', '需輔導'],
                            yticklabels=['通過', '需輔導'])

                if MY_FONT:
                    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), fontproperties=MY_FONT, fontsize=11)
                    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), fontproperties=MY_FONT, fontsize=11)
                    ax_cm.set_ylabel('真實情況', fontproperties=MY_FONT, fontsize=12)
                    ax_cm.set_xlabel('模型判斷', fontproperties=MY_FONT, fontsize=12)
                st.pyplot(fig_cm)

            st.divider()

            st.subheader("🔮 單一學生即時診斷系統")
            st.markdown("請輸入學生當前的狀態參數，AI 將即時評估其學習失敗的風險機率。")

            with st.container(border=True):
                col_in1, col_in2, col_in3, col_in4 = st.columns(4)

                with col_in1:
                    in_lag = st.number_input("練習延遲 (小時)", min_value=0.0, max_value=24.0, value=2.0, step=0.5)
                with col_in2:
                    if col_difficulty in df.columns:
                        diff_opts = sorted(df[col_difficulty].astype(str).unique())
                        idx = diff_opts.index('中') if '中' in diff_opts else 0
                        in_diff = st.selectbox("題目難度", diff_opts, index=idx)
                    else:
                        in_diff = st.selectbox("題目難度", ["未知"])
                with col_in3:
                    score_max = df[col_score].max()
                    if score_max > 1.0:
                        in_ability = st.slider("學生平均程度 (0-100)", 0, 100, 80)
                    else:
                        in_ability = st.slider("學生平均程度 (0-1)", 0.0, 1.0, 0.8)
                with col_in4:
                    in_duration = st.number_input("答題耗時 (秒)", min_value=1, max_value=600, value=60)

                predict_btn = st.button("🔍 進行診斷分析", type="primary", use_container_width=True)

            if predict_btn:
                try:
                    le = st.session_state['label_encoder']
                    try:
                        diff_val = le.transform([str(in_diff)])[0]
                    except:
                        diff_val = 0

                    input_vector = [[in_lag, diff_val, in_ability, in_duration]]
                    pred_prob = clf.predict_proba(input_vector)[0][1]

                    if pred_prob > 0.5:
                        st.markdown(f"""
                            <div class="prediction-result pred-danger">
                                <h3 style='margin:0'>🔴 高風險警示</h3>
                                <div style='font-size:3rem; font-weight:bold;'>{pred_prob:.1%}</div>
                                <p>失敗機率極高</p>
                                <hr style='border-color: rgba(255,255,255,0.3);'>
                                <p style='text-align:left; font-size:0.9rem;'>💡 <b>建議介入：</b> 此學生可能尚未精熟，建議立即提供補救教學或提示。</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="prediction-result pred-safe">
                                <h3 style='margin:0'>🟢 學習狀況良好</h3>
                                <div style='font-size:3rem; font-weight:bold;'>{pred_prob:.1%}</div>
                                <p>失敗機率低</p>
                                <hr style='border-color: rgba(255,255,255,0.3);'>
                                <p style='text-align:left; font-size:0.9rem;'>💡 <b>建議策略：</b> 學生掌握度高，可維持當前學習節奏。</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"預測時發生錯誤: {e}")


if __name__ == "__main__":
    main()