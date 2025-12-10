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
# 0. 全域設定與 Liquid Glass CSS 美化
# ==============================================================================

st.set_page_config(
    page_title="教育大數據｜學習歷程分析儀表板",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Liquid Glass 專業級 CSS 樣式 ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');

    /* 1. 全局背景：液態漸層 */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        background-attachment: fixed;
        color: #E0E0E0;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* 2. 字體設定 */
    html, body, [class*="css"] {
        font-family: 'Noto Sans TC', sans-serif;
    }

    h1, h2, h3 {
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }

    /* 3. Sidebar 玻璃化 */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 4. 主標題樣式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0px 0px 10px rgba(79, 172, 254, 0.5));
    }

    .sub-title {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* 5. Liquid Glass 卡片 (通用) */
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }

    .glass-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 255, 255, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }

    /* KPI 專用樣式 */
    .kpi-title { 
        font-size: 0.9rem; 
        color: rgba(255, 255, 255, 0.6); 
        text-transform: uppercase; 
        letter-spacing: 1px;
        margin-bottom: 5px; 
    }
    .kpi-value { 
        font-size: 2.2rem; 
        font-weight: 700; 
        color: #FFFFFF;
        text-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    .kpi-note { 
        font-size: 0.85rem; 
        color: #00f2fe; 
        font-weight: 500; 
        margin-top: 5px;
    }

    /* 預測結果卡片 */
    .prediction-result {
        padding: 25px; 
        border-radius: 20px; 
        color: white; 
        text-align: center; 
        margin-top: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    .pred-danger { 
        background: linear-gradient(135deg, rgba(255, 65, 108, 0.8) 0%, rgba(255, 75, 43, 0.8) 100%); 
    }
    .pred-safe { 
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.8) 0%, rgba(56, 239, 125, 0.8) 100%); 
    }

    /* Streamlit 元件優化 (按鈕、輸入框) */
    .stButton>button {
        border-radius: 12px;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border: none;
        color: #0F3460;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0, 242, 254, 0.6);
        color: white;
    }

    /* Tabs 優化 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.05);
        border-radius: 10px;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255,255,255,0.2) !important;
        border-color: #00f2fe !important;
        color: #00f2fe !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.05) !important;
        color: white !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        backdrop-filter: blur(10px);
        color: white !important;
    }

    </style>
    """, unsafe_allow_html=True)


def get_chinese_font():
    """獲取中文字體"""
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
        paths = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"]
        for p in paths:
            if os.path.exists(p):
                return font_manager.FontProperties(fname=p)
    return None


def set_plot_style_dark():
    """
    設定全域繪圖風格 - 暗色玻璃模式
    讓圖表背景透明，文字變白
    """
    # 使用 ticks 風格，減少 seaborn 預設背景干擾
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    # Matplotlib 全域設定
    plt.rcParams['figure.facecolor'] = 'none'  # 圖表背景全透明
    plt.rcParams['axes.facecolor'] = 'none'  # 座標軸背景全透明
    plt.rcParams['savefig.facecolor'] = 'none'

    # 文字顏色設定為白色/亮灰
    text_color = '#E0E0E0'
    plt.rcParams['text.color'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color

    # 【修正點】使用 Tuple (R, G, B, Alpha) 設定半透明顏色，而不是 CSS 字串
    # 白色 (1,1,1) 透明度 0.3
    plt.rcParams['axes.edgecolor'] = (1, 1, 1, 0.3)
    # 白色 (1,1,1) 透明度 0.1
    plt.rcParams['grid.color'] = (1, 1, 1, 0.1)

    my_font = get_chinese_font()
    if my_font:
        plt.rcParams['font.sans-serif'] = [my_font.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        sns.set(font=my_font.get_name())

        # sns.set 有時會重置顏色，這裡再次強制覆蓋
        plt.rcParams['figure.facecolor'] = 'none'
        plt.rcParams['axes.facecolor'] = 'none'
        plt.rcParams['text.color'] = text_color
        plt.rcParams['axes.labelcolor'] = text_color
        plt.rcParams['xtick.color'] = text_color
        plt.rcParams['ytick.color'] = text_color
        plt.rcParams['axes.edgecolor'] = (1, 1, 1, 0.3)
        plt.rcParams['grid.color'] = (1, 1, 1, 0.1)

        return my_font
    return None


MY_FONT = set_plot_style_dark()


def display_kpi_card(title, value, note, color_border="#4F8BF9"):
    # 在 CSS 中已定義 .glass-card，這裡改用純 CSS 結構，移除行內 style
    # 為了保持原有的頂部邊框顏色效果，我們使用 border-top
    st.markdown(f"""
    <div class="glass-card" style="border-top: 3px solid {color_border}; text-align: center;">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# 1. 資料處理核心邏輯 (0-24H 高密度版) - 保持不變
# ==============================================================================

@st.cache_data(show_spinner="🚀 正在讀取並分析固定路徑資料...")
def load_and_preprocess_data(file_path, remove_outliers=False):
    stats = {}
    try:
        df = pd.read_csv(file_path)
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
            st.error("無有效資料。")
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

    # --- Sidebar (Glass Style) ---
    with st.sidebar:
        st.title("🎛️ 控制台")
        st.markdown("---")
        st.info("📂 資料來源：教育大數據競賽")

        st.markdown("### ⚙️ 參數設定")
        enable_outlier_removal = st.toggle("IQR 極端值過濾", value=True)
        st.caption("ℹ️ 分析範圍：0 ~ 24 小時")
        st.markdown("---")

    # --- Header ---
    st.markdown('<div class="main-title">🎓 教育大數據：學習黃金窗口</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">24H Learning Consolidation Analytics Dashboard</div>', unsafe_allow_html=True)

    # --- GitHub 部署路徑設定 ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 根據你的需求修改路徑
    FILE_PATH = os.path.join(current_dir, 'resource', 'anonymized_file0115.csv')

    # 測試用: 如果上一層找不到，試試看當前目錄 (方便本地測試)
    if not os.path.exists(FILE_PATH):
        alt_path = os.path.join(current_dir, 'anonymized_file0115.csv')
        if os.path.exists(alt_path):
            FILE_PATH = alt_path

    if not os.path.exists(FILE_PATH):
        st.error(f"❌ 找不到資料檔案")
        st.warning(f"系統嘗試讀取的路徑是： `{FILE_PATH}`")
        return

    # --- 讀取檔案 ---
    df, median_score, stats = load_and_preprocess_data(FILE_PATH, enable_outlier_removal)

    if df is None: return

    col_score = '首次答題正確率'
    col_duration = '首次答題時間（秒）'
    col_difficulty = '難易度'
    col_user = '學生姓名去識別化'

    # --- KPI Section (Glass Cards) ---
    avg_score = df[col_score].mean()
    score_fmt = f"{avg_score * 100:.1f}%" if avg_score <= 1.0 else f"{avg_score:.1f}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        display_kpi_card("有效樣本 (24H)", f"{len(df):,}", f"保留率: {len(df) / stats['original_count']:.1%}",
                         "#00c6ff")
    with col2:
        display_kpi_card("不重複學生", f"{df[col_user].nunique():,}", "Active Learners", "#9b59b6")
    with col3:
        display_kpi_card("平均正確率", score_fmt, "Overall Accuracy", "#00f2fe")
    with col4:
        display_kpi_card("平均答題耗時", f"{df[col_duration].mean():.1f}s", "Avg Duration", "#f7b731")

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
                <div class="glass-card">
                    <h4 style="margin-top:0; color: #00f2fe;">🧹 清理摘要</h4>
                    <p><b>原始筆數：</b> {stats['original_count']:,}</p>
                    <p><b>24H樣本：</b> {stats['valid_24h_count']:,}</p>
                    <p><b>最終分析：</b> {stats['final_count']:,}</p>
                    <hr style="border-top: 1px solid rgba(255,255,255,0.1);">
                    <small style="opacity:0.6;">排除條件：時間異常、數值錯誤、IQR極端值。</small>
                </div>
                """, unsafe_allow_html=True)

    # Helper function for transparent plots
    def create_glass_figure(figsize=(10, 5)):
        fig, ax = plt.subplots(figsize=figsize)
        # 強制透明
        fig.patch.set_alpha(0.0)
        ax.set_facecolor((0, 0, 0, 0))
        # 網格線微調
        ax.grid(True, linestyle='--', alpha=0.15, color='white')
        return fig, ax

    # Tab 2: 鞏固曲線
    with tab2:
        st.subheader("📉 記憶鞏固趨勢分析")
        col_ctrl1, col_ctrl2 = st.columns([1, 3])
        with col_ctrl1:
            # 直接放控制項即可，CSS 會自動處理輸入框的樣式
            y_opt = st.selectbox("分析指標 (Y軸)", [col_score, '擷取訊息正確率', '發展解釋正確率'], key="tab2_y_opt")
            split_diff = st.toggle("依難易度分層", value=True, key="tab2_diff_toggle")

        with col_ctrl2:
            fig, ax = create_glass_figure()

            if split_diff and col_difficulty in df.columns:
                diff_order = ['易', '中', '難']
                # 霓虹配色
                colors = {'易': '#00ff87', '中': '#ffdd00', '難': '#ff0055'}
                present_diffs = [d for d in diff_order if d in df[col_difficulty].unique()]
                agg = df.groupby(['lag_bin_mid', col_difficulty])[y_opt].mean().reset_index()
                agg = agg[agg[col_difficulty].isin(present_diffs)]
                sns.lineplot(data=agg, x='lag_bin_mid', y=y_opt, hue=col_difficulty,
                             hue_order=present_diffs, palette=colors,
                             marker='o', linewidth=2.5, ax=ax)
                # 設定 Legend 文字顏色
                if ax.legend_:
                    plt.setp(ax.get_legend().get_texts(), color='#E0E0E0', fontproperties=MY_FONT)
            else:
                agg = df.groupby('lag_bin_mid')[y_opt].mean().reset_index()
                sns.lineplot(data=agg, x='lag_bin_mid', y=y_opt, marker='o',
                             color='#00c6ff', linewidth=3, label="全體平均", ax=ax, fontproperties=MY_FONT)
                if ax.legend_:
                    plt.setp(ax.get_legend().get_texts(), color='#E0E0E0')

            ax.set_xticks(np.arange(0, 25, 3))
            ax.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
            ax.set_ylabel("平均分數", fontproperties=MY_FONT)
            ax.set_title(f"24小時內 {y_opt} 變化趨勢", fontproperties=MY_FONT, fontsize=14, color='white')

            st.pyplot(fig)

        with st.expander("💡 圖表解讀"):
            st.markdown("""
            *   **趨勢意義**：觀察曲線是否隨時間上升。若上升，代表存在「記憶鞏固」效應。
            *   **難度分層**：「難」的任務在初期正確率較低，但若經過適當延遲，其回升幅度可能更明顯。
            """)

    # Tab 3: 認知負荷
    with tab3:
        st.subheader("⏱️ 認知負荷 (答題時間) 分析")
        col_t1, col_t2 = st.columns([1, 3])
        with col_t1:
            split_time_diff = st.toggle("依難易度分層", value=True, key="time_split")

        with col_t2:
            fig2, ax2 = create_glass_figure()

            if split_time_diff and col_difficulty in df.columns:
                diff_order = ['易', '中', '難']
                colors = {'易': '#00ff87', '中': '#ffdd00', '難': '#ff0055'}
                present_diffs = [d for d in diff_order if d in df[col_difficulty].unique()]
                agg = df.groupby(['lag_bin_mid', col_difficulty])[col_duration].median().reset_index()
                agg = agg[agg[col_difficulty].isin(present_diffs)]
                sns.lineplot(data=agg, x='lag_bin_mid', y=col_duration, hue=col_difficulty, hue_order=present_diffs,
                             palette=colors, marker='s', linewidth=2.5, ax=ax2)
                if ax2.legend_:
                    plt.setp(ax2.get_legend().get_texts(), color='#E0E0E0')
            else:
                agg = df.groupby('lag_bin_mid')[col_duration].median().reset_index()
                ax2.fill_between(agg['lag_bin_mid'], agg[col_duration], color="#ffdd00", alpha=0.1)
                sns.lineplot(data=agg, x='lag_bin_mid', y=col_duration, marker='s', color='#ffdd00', linewidth=2.5,
                             label="全體中位數", ax=ax2)
                if ax2.legend_:
                    plt.setp(ax2.get_legend().get_texts(), color='#E0E0E0')

            ax2.set_xticks(np.arange(0, 25, 3))
            ax2.set_xlabel("練習延遲時間 (小時)", fontproperties=MY_FONT)
            ax2.set_ylabel("答題時間中位數 (秒)", fontproperties=MY_FONT)

            st.pyplot(fig2)

        with st.expander("💡 圖表解讀"):
            st.markdown("""
            *   **費力提取**：若長時間延遲後，答題時間顯著增加且正確率未下降，代表學生正在進行「費力提取」，有助於長期記憶。
            """)

    # Tab 4: 分群差異
    with tab4:
        st.subheader("👥 學習者分群行為差異")
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.markdown("##### 1. 高分組 vs 潛力組 (正確率)")
            fig3, ax3 = create_glass_figure(figsize=(6, 5))

            for g, c in zip(['高分組', '潛力組'], ['#00c6ff', '#ff0055']):
                sub = df[df['ability_group'] == g]
                agg = sub.groupby('lag_bin_mid')[col_score].mean().reset_index()
                sns.lineplot(data=agg, x='lag_bin_mid', y=col_score, marker='o', label=g, color=c, linewidth=2, ax=ax3)

            ax3.set_xticks(np.arange(0, 25, 6))
            ax3.set_xlabel("小時", fontproperties=MY_FONT)
            ax3.set_ylabel("平均正確率", fontproperties=MY_FONT)
            if ax3.legend_: plt.setp(ax3.get_legend().get_texts(), color='#E0E0E0')
            st.pyplot(fig3)

        with col_d2:
            st.markdown("##### 2. 知識向度差異")
            candidate_cols = ['擷取訊息正確率', '發展解釋正確率', '廣泛理解正確率', '文本形式正確率', '文本理解正確率']
            valid_options = [c for c in candidate_cols if c in df.columns]
            know_cols = st.multiselect("請選擇向度:", options=valid_options,
                                       default=[valid_options[0]] if valid_options else None, key="tab4_know_cols")

            fig4, ax4 = create_glass_figure(figsize=(6, 5))
            if know_cols:
                # 鮮豔的顏色循環
                palette = sns.color_palette("husl", len(know_cols))
                markers = ['o', 's', '^', 'D', 'v']
                for idx, col in enumerate(know_cols):
                    agg = df.groupby('lag_bin_mid')[col].mean().reset_index()
                    label_name = col.replace('正確率', '')
                    sns.lineplot(data=agg, x='lag_bin_mid', y=col, marker=markers[idx % 5], label=label_name,
                                 color=palette[idx], linewidth=2, ax=ax4)
                ax4.set_xticks(np.arange(0, 25, 6))
                ax4.set_xlabel("小時", fontproperties=MY_FONT)
                ax4.set_ylabel("平均正確率", fontproperties=MY_FONT)
                if ax4.legend_: plt.setp(ax4.get_legend().get_texts(), color='#E0E0E0')
                st.pyplot(fig4)
            else:
                st.info("請選擇向度")

    # Tab 5: AI 預測
    with tab5:
        st.subheader("🤖 AI 學習風險預測模型")

        def train_model_callback():
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
            else:
                st.error("樣本不足")

        col_train_btn, _ = st.columns([1, 4])
        with col_train_btn:
            st.button("🚀 訓練模型", type="primary", use_container_width=True, on_click=train_model_callback)

        if st.session_state['trained_model'] is not None:
            st.success(f"模型已就緒！準確率: {st.session_state['accuracy']:.2%}")

            col_plot1, col_plot2 = st.columns(2)
            clf = st.session_state['trained_model']
            feats = st.session_state['model_features']

            with col_plot1:
                st.markdown("##### 🔑 影響因子權重")
                name_mapping = {'lag_hours': '練習延遲時間', 'diff_code': '任務難易度', 'user_ability': '學生程度',
                                col_duration: '答題耗時'}
                imp = pd.Series(clf.feature_importances_, index=feats).sort_values()
                imp.index = [name_mapping.get(x, x) for x in imp.index]

                fig_imp, ax_imp = create_glass_figure(figsize=(6, 4))

                # Barh color
                imp.plot(kind='barh', ax=ax_imp, color='#00f2fe', width=0.7)

                # 調整 y 軸標籤顏色
                if MY_FONT:
                    ax_imp.set_yticklabels(imp.index, fontproperties=MY_FONT, fontsize=11, color='white')
                    ax_imp.set_xlabel("Importance", fontproperties=MY_FONT, color='white')
                st.pyplot(fig_imp)

            with col_plot2:
                st.markdown("##### 🔍 預測混淆矩陣")
                cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
                fig_cm, ax_cm = create_glass_figure(figsize=(6, 4))

                # Heatmap with dark background friendly colors
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False,
                            alpha=0.8,  # <--- 加入這行，讓色塊稍微透明
                            xticklabels=['通過', '需輔導'], yticklabels=['通過', '需輔導'])

                # 文字顏色調整
                for t in ax_cm.texts: t.set_color('white')

                if MY_FONT:
                    ax_cm.set_xticklabels(ax_cm.get_xticklabels(), fontproperties=MY_FONT, fontsize=11, color='white')
                    ax_cm.set_yticklabels(ax_cm.get_yticklabels(), fontproperties=MY_FONT, fontsize=11, color='white')
                    ax_cm.set_ylabel('真實情況', fontproperties=MY_FONT, fontsize=12, color='white')
                    ax_cm.set_xlabel('模型判斷', fontproperties=MY_FONT, fontsize=12, color='white')
                st.pyplot(fig_cm)

            with st.expander("💡 模型洞察：什麼決定了學習成敗？", expanded=True):
                st.markdown("""
                **1. 關鍵影響因子**
                *   **學生程度**：預測力最強，反映「過去表現」對未來的影響。
                *   **練習延遲**：在排除資質後，**「時間」是影響成敗最重要的可控變因**。

                **2. 應用價值**
                *   只要能準確抓出高風險學生，系統就能及時發出警示，發揮「學習安全網」功能。
                """)

            st.divider()
            st.subheader("🔮 單一學生即時診斷")
            with st.container():
                # 這裡可以改用 st.container 本身，不需要外包 HTML
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    in_lag = st.number_input("練習延遲 (H)", 0.0, 24.0, 2.0, 0.5)
                with c2:
                    diff_opts = sorted(df[col_difficulty].astype(str).unique()) if col_difficulty in df.columns else [
                        "未知"]
                    idx = diff_opts.index('中') if '中' in diff_opts else 0
                    in_diff = st.selectbox("題目難度", diff_opts, index=idx)
                with c3:
                    s_max = df[col_score].max()
                    in_ability = st.slider("學生程度", 0, 100, 80) if s_max > 1.0 else st.slider("學生程度", 0.0, 1.0,
                                                                                                 0.8)
                with c4:
                    in_duration = st.number_input("耗時 (秒)", 1, 600, 60)

                if st.button("🔍 診斷", type="primary", use_container_width=True):
                    try:
                        d_val = st.session_state['label_encoder'].transform([str(in_diff)])[0] if st.session_state[
                            'label_encoder'] else 0
                    except:
                        d_val = 0
                    prob = clf.predict_proba([[in_lag, d_val, in_ability, in_duration]])[0][1]

                    if prob > 0.5:
                        st.markdown(
                            f"<div class='prediction-result pred-danger'><h3>🔴 高風險</h3><h1>{prob:.1%}</h1><p>建議：立即介入輔導</p></div>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div class='prediction-result pred-safe'><h3>🟢 狀況良好</h3><h1>{prob:.1%}</h1><p>建議：保持當前學習步調</p></div>",
                            unsafe_allow_html=True)


if __name__ == "__main__":
    main()