# ==============================================================================
# 2025 教育大數據競賽：完整分析腳本 (最終定稿版)
# 隊名：三三三旅
# 功能：EDA、ANOVA 檢定、Random Forest 預測與應用模擬
# ==============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 環境設定與資料讀取
# ==============================================================================
print("🚀 [Step 1] 初始化與讀取資料...")

# --- 設定字體 (解決中文亂碼) ---
font_file = "../resource/NotoSansTC-Regular.ttf"
if os.path.exists(font_file):
    my_font = font_manager.FontProperties(fname=font_file)
    # 設定全域字體
    plt.rcParams['font.sans-serif'] = ['Noto Sans TC']
    plt.rcParams['axes.unicode_minus'] = False
    print("✅ 字體載入成功。")
else:
    my_font = None
    print("⚠️ 警告：找不到中文字體檔，圖表文字可能無法正常顯示。")

# --- 欄位名稱對照 (依據您的資料) ---
FILE_PATH = '../resource/anonymized_file0115.csv'
COL_USER = '學生姓名去識別化'
COL_START_TIME = '任務派發時間'
COL_END_TIME = '學生首次送出答案的時間點'
COL_SCORE = '首次答題正確率'
COL_DURATION = '首次答題時間（秒）'
COL_DIFFICULTY = '難易度'
COL_TASK_NAME = '任務名稱'

try:
    df = pd.read_csv(FILE_PATH)
    print(f"✅ 讀取成功，原始資料共 {len(df)} 筆。")

    # --- 資料清理與特徵工程 ---
    # 1. 時間格式轉換
    df[COL_START_TIME] = pd.to_datetime(df[COL_START_TIME], errors='coerce')
    df[COL_END_TIME] = pd.to_datetime(df[COL_END_TIME], errors='coerce')

    # 2. 移除無效資料 (排除時間缺失或非數值資料)
    df = df.dropna(subset=[COL_START_TIME, COL_END_TIME, COL_SCORE, COL_DURATION])

    # 3. 計算時間延遲 (Practice Lag)
    df['lag_hours'] = (df[COL_END_TIME] - df[COL_START_TIME]) / pd.Timedelta(hours=1)

    # 4. 關鍵區間鎖定 (0~168小時，排除極端值與重複作答干擾)
    df_final = df[(df['lag_hours'] >= 0) & (df['lag_hours'] <= 168)].copy()

    # 5. 確保數值格式
    df_final[COL_SCORE] = pd.to_numeric(df_final[COL_SCORE], errors='coerce')
    df_final[COL_DURATION] = pd.to_numeric(df_final[COL_DURATION], errors='coerce')

    print(f"✅ 清理完成，剩餘 {len(df_final)} 筆有效分析資料。")

    # --- 學習者能力分群 (Learner Grouping) ---
    # 使用中位數切分法 (Median Split)
    user_stats = df_final.groupby(COL_USER)[COL_SCORE].mean()
    median_score = user_stats.median()


    def get_group(user_id):
        s = user_stats.get(user_id)
        if s is None: return '未知'
        return '高分組' if s >= median_score else '潛力組'


    df_final['student_group'] = df_final[COL_USER].apply(get_group)
    print(f"ℹ️ 學生分群基準 (中位數): {median_score:.2f}")

    # --- 對數分箱 (Log Binning) ---
    # 用於視覺化學習初期的細微變化
    log_bins = [0] + list(np.logspace(0, 7, num=8, base=2)) + [168]
    log_labels = [f'{log_bins[i]:.1f}-{log_bins[i + 1]:.1f}h' for i in range(len(log_bins) - 1)]
    df_final['log_lag_bin'] = pd.cut(df_final['lag_hours'], bins=log_bins, labels=log_labels, right=False)

except Exception as e:
    print(f"❌ 資料處理發生錯誤: {e}")
    df_final = None

# ==============================================================================
# 2. 視覺化 (EDA) - 圖 F-1 & F-2
# ==============================================================================
if df_final is not None:
    print("\n🚀 [Step 2] 繪製探索性分析圖表...")
    sns.set_style("whitegrid")

    # --- 圖 F-1: 正確率 ---
    plt.figure(figsize=(16, 8))
    ax1 = sns.barplot(
        data=df_final, x='log_lag_bin', y=COL_SCORE, hue='student_group',
        palette={'高分組': 'dodgerblue', '潛力組': 'salmon'}, errorbar=('ci', 95)
    )
    plt.title('圖 F-1: 不同能力組在「對數時間延遲」下的平均正確率 (CI=95%)', fontsize=16, fontproperties=my_font)
    plt.xlabel('任務派發到首次作答的間隔 (小時)', fontsize=12, fontproperties=my_font)
    plt.ylabel('首次答題正確率', fontsize=12, fontproperties=my_font)
    plt.xticks(rotation=45)

    # 修復圖例亂碼
    if my_font:
        L = plt.legend(title='學生分組', prop=my_font)
        plt.setp(L.get_title(), fontproperties=my_font)
    plt.show()

    # --- 圖 F-2: 答題耗時 (核心亮點) ---
    plt.figure(figsize=(16, 8))
    ax2 = sns.barplot(
        data=df_final, x='log_lag_bin', y=COL_DURATION, hue='student_group',
        palette={'高分組': 'dodgerblue', '潛力組': 'salmon'}, errorbar=('ci', 95)
    )
    plt.title('圖 F-2: 不同能力組在「對數時間延遲」下的平均答題耗時 (秒)', fontsize=16, fontproperties=my_font)
    plt.xlabel('任務派發到首次作答的間隔 (小時)', fontsize=12, fontproperties=my_font)
    plt.ylabel('平均首次答題時間 (秒)', fontsize=12, fontproperties=my_font)
    plt.xticks(rotation=45)

    # 修復圖例亂碼
    if my_font:
        L = plt.legend(title='學生分組', prop=my_font)
        plt.setp(L.get_title(), fontproperties=my_font)
    plt.show()

# ==============================================================================
# 3. 統計檢定 (ANOVA)
# ==============================================================================
if df_final is not None:
    print("\n🚀 [Step 3] 執行雙因子變異數分析 (Two-way ANOVA)...")

    # 準備乾淨的資料給 statsmodels
    df_stat = df_final.copy()
    df_stat = df_stat.rename(columns={
        COL_DURATION: 'Duration',
        COL_SCORE: 'Score',
        'student_group': 'Group',
        'log_lag_bin': 'TimeBin'
    })

    # 檢定答題時間 (Duration) 的交互作用
    model_duration = ols('Duration ~ C(Group) + C(TimeBin) + C(Group):C(TimeBin)', data=df_stat).fit()
    anova_table = anova_lm(model_duration, typ=2)
    p_val = anova_table.loc['C(Group):C(TimeBin)', 'PR(>F)']
    f_val = anova_table.loc['C(Group):C(TimeBin)', 'F']

    print("\n📊 答題時間 ANOVA 結果 (交互作用):")
    print(f"   F-Value: {f_val:.2f}")
    print(f"   P-Value: {p_val:.4e}")
    if p_val < 0.05:
        print("✅ 結果：顯著！證實高分組與潛力組的行為模式具備統計顯著差異。")
    else:
        print("⚠️ 結果：未達顯著水準。")

# ==============================================================================
# 4. AI 預測模型與應用 (Random Forest V2.0)
# ==============================================================================
if df_final is not None:
    print("\n🚀 [Step 4] 啟動特徵工程與模型優化 (V2.0)...")

    # --- 4.1 進階特徵工程 (Advanced Feature Engineering) ---
    df_model = df_final.copy()

    # 特徵 A: 學生基礎能力 (User Ability)
    df_model['user_ability'] = df_model.groupby(COL_USER)[COL_SCORE].transform('mean')

    # 特徵 B: 題目真實難度 (Real Task Difficulty)
    df_model['task_pass_rate'] = df_model.groupby(COL_TASK_NAME)[COL_SCORE].transform('mean')

    # 特徵 C: 時間的非線性變換 (Log Lag)
    df_model['log_lag'] = np.log1p(df_model['lag_hours'])

    # 特徵 D: 原始難度編碼
    le = LabelEncoder()
    df_model['diff_code'] = le.fit_transform(df_model[COL_DIFFICULTY].astype(str))

    # --- 4.2 定義目標與訓練 ---
    # 定義目標：需要幫助 (正確率 < 80)
    target_threshold = 80 if df_final[COL_SCORE].max() > 1 else 0.8
    df_model['need_help'] = np.where(df_model[COL_SCORE] < target_threshold, 1, 0)

    feature_cols = ['lag_hours', 'log_lag', 'diff_code', 'user_ability', 'task_pass_rate']
    df_model = df_model.dropna(subset=feature_cols)

    X = df_model[feature_cols]
    y = df_model['need_help']

    # 切分資料 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 訓練隨機森林 (參數優化)
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 預測
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    # --- 4.3 繪製圖 H-1: 混淆矩陣 (綠色版) ---
    cm = confusion_matrix(y_test, y_pred)

    # 計算 Recall (召回率)
    tp = cm[1, 1]
    fn = cm[1, 0]
    recall = tp / (tp + fn)
    print(f"📊 模型召回率 (Recall): {recall:.2%} (成功捕捉高風險學生的比例)")

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                     xticklabels=['預測:通過', '預測:需幫助'],
                     yticklabels=['實際:通過', '實際:需幫助'])
    plt.title('圖 H-1 (優化版): 預測模型混淆矩陣', fontsize=16, fontproperties=my_font)
    plt.xlabel('模型預測', fontsize=12, fontproperties=my_font)
    plt.ylabel('真實情況', fontsize=12, fontproperties=my_font)
    # 強制設定軸刻度字體
    plt.xticks(fontproperties=my_font)
    plt.yticks(fontproperties=my_font)
    plt.show()

    # --- 4.4 繪製圖 H-2: ROC 曲線 ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label=f'Optimized Model (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.title('圖 H-2 (優化版): 模型鑑別力分析', fontsize=16, fontproperties=my_font)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right")
    plt.show()

    # --- 4.5 繪製圖 G-1: 特徵重要性 ---
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_cols[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
    plt.title('圖 G-1 (優化版): 關鍵影響因子排名', fontsize=16, fontproperties=my_font)
    plt.xlabel('重要性權重', fontsize=12, fontproperties=my_font)
    # y軸標籤可能需要說明，這裡保持英文變數名以免亂碼，但可在報告中解釋
    plt.show()

    # ==========================================================================
    # 5. 應用模擬 (Scenario Demo)
    # ==========================================================================
    print("\n🚀 [Step 5] 智慧複習系統：情境模擬 Demo")
    print("-" * 60)

    # 5.1 自動搜尋最佳範例 (尋找高風險的潛力學生)
    df_search = df_model.copy()
    df_search['risk_prob'] = clf.predict_proba(X)[:, 1]

    # 條件：中等程度學生(0.4~0.7) + 拖延超過2天 + 風險極高
    target_group = df_search[
        (df_search['user_ability'] >= 0.4) &
        (df_search['user_ability'] <= 0.7) &
        (df_search['lag_hours'] > 48)
        ].sort_values(by='risk_prob', ascending=False)

    if len(target_group) > 0:
        case_study = target_group.iloc[0]
        # 還原難度文字
        try:
            real_diff_text = le.inverse_transform([int(case_study['diff_code'])])[0]
        except:
            real_diff_text = str(case_study['diff_code'])

        print(f"🔥 情境 B (危險 - 遺忘警示):")
        print(f"   - 任務難度: 「{real_diff_text}」")
        print(f"   - 學生程度: {case_study['user_ability'] * 100:.1f} 分 (中等程度)")
        print(f"   - 延遲時間: {case_study['lag_hours']:.1f} 小時 (約 {case_study['lag_hours'] / 24:.1f} 天)")
        print(f" -> AI 預測失敗風險: {case_study['risk_prob'] * 100:.1f}%")
        print("🔴 系統建議：【立即複習】(偵測到遺忘風險飆升)")
    else:
        print("⚠️ 未找到極端高風險案例，請參考圖表結果。")
    print("-" * 60)

print("\n✅ 所有分析執行完畢！祝比賽順利！")