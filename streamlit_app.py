"""
Streamlit application to visualise findings from the
"學習的悖論：探勘適度困難與記憶鞏固對不同學習者與知識類型影響的數據驅動研究"。

This app is intended to showcase the key insights described in
the accompanying proposal (see README) and to provide an
interactive playground where readers can explore synthetic
data that emulate the main phenomena uncovered in the
project.  In particular the underlying study discovered a
pronounced U‑shaped relationship between the time delay
before practising a newly learned concept and the eventual
performance on the first follow‑up exercise.  A peak in
performance was observed around 6–8 hours after first
exposure, giving rise to a so‑called “golden consolidation
window”【432059056011258†L162-L164】.  Differences among learner
ability groups and knowledge types were also found, as well
as rising cognitive load (response time) with longer delays
【432059056011258†L165-L177】.

The synthetic dataset used here is not derived from the
competition data; it is generated on the fly to mimic the
patterns described in the proposal.  Feel free to adjust
parameters or replace it with your own data.

Additional professional insights about memory consolidation,
desirable difficulties and the spacing effect are provided
throughout the app to help contextualise the findings.

To run this app you will need to install `streamlit` and
`pandas`, `numpy`, `matplotlib` and `scikit‑learn`.  Once these
packages are installed you can launch the app by running:

    streamlit run streamlit_app.py

"""

import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image
import io


def generate_synthetic_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate a synthetic dataset that emulates the U‑shaped
    practice‑lag effect observed in the study.

    Parameters
    ----------
    n_samples: int
        Number of synthetic learner–task observations to generate.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:
        - practice_lag: time delay between learning and first practice (hours)
        - ability: categorical learner ability group (High, Medium, Low)
        - knowledge_type: categorical type (Fact, Concept)
        - accuracy: simulated first‑attempt correctness (float)
        - response_time: simulated reaction time (seconds)
        - label: binary version of accuracy (0/1) used for modelling
    """
    rng = np.random.default_rng(42)
    # Practice lag between 0 and 24 hours
    practice_lag = rng.uniform(0, 24, n_samples)

    # Assign learners to ability groups with unequal probabilities
    ability = rng.choice(["High", "Medium", "Low"], size=n_samples, p=[0.3, 0.5, 0.2])

    # Assign knowledge types (Fact vs Concept); concept corresponds to higher‑order understanding
    knowledge_type = rng.choice(["Fact", "Concept"], size=n_samples, p=[0.6, 0.4])

    # Define base performance as a U‑shaped function of practice lag
    # Peak around 7 hours; use a parabola scaled between 0.4 and 0.9
    base_accuracy = 0.4 + 0.5 * np.exp(-((practice_lag - 7) ** 2) / (2 * 3.0 ** 2))

    # Adjust performance based on ability
    ability_factor = np.where(ability == "High", 0.1, np.where(ability == "Low", -0.1, 0.0))

    # Adjust performance based on knowledge type
    type_factor = np.where(knowledge_type == "Concept", 0.05, -0.05)

    # Add some noise
    noise = rng.normal(0, 0.03, n_samples)

    accuracy = np.clip(base_accuracy + ability_factor + type_factor + noise, 0, 1)

    # Simulate response time; longer delay leads to longer responses
    response_time = 5 + 5 * (practice_lag / 24) + rng.normal(0, 0.5, n_samples)

    # Binary label: success if accuracy > 0.5
    label = (accuracy > 0.5).astype(int)

    return pd.DataFrame({
        "practice_lag": practice_lag,
        "ability": ability,
        "knowledge_type": knowledge_type,
        "accuracy": accuracy,
        "response_time": response_time,
        "label": label,
    })


def train_random_forest(df: pd.DataFrame) -> dict:
    """Train a simple random forest classifier on the synthetic data.

    The model predicts whether a learner will answer a practice question
    correctly (binary outcome) based on practice lag, ability,
    knowledge type and task difficulty.  Feature importance values
    returned here are merely illustrative.

    Returns
    -------
    dict
        Dictionary containing the fitted model, metrics and feature importances.
    """
    # Encode categorical features manually
    X = df[["practice_lag"]].copy()
    # One‑hot encoding for ability
    for group in ["High", "Medium", "Low"]:
        X[f"ability_{group}"] = (df["ability"] == group).astype(int)
    # One‑hot encoding for knowledge type
    for ktype in ["Fact", "Concept"]:
        X[f"type_{ktype}"] = (df["knowledge_type"] == ktype).astype(int)

    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    importances = model.feature_importances_
    importance_dict = {feature: imp for feature, imp in zip(X.columns, importances)}
    return {
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "importances": importance_dict,
    }


def main() -> None:
    """Main function to build the Streamlit app."""
    st.set_page_config(
        page_title="記憶鞏固與適度困難分析",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("學習悖論分析儀表板")
    st.markdown(
        """
        ## 背景與研究動機

        本專案旨在揭示 **「適度困難」(Desirable Difficulty)** 理論在真實數位學習情境中的表現。研究團隊透過挖掘學習歷程大數據，發現「記憶鞏固」存在黃金時間窗口：
        **學習後約 6–8 小時進行首次練習能顯著提升答對率**\[1]。更有趣的是，不同能力組
        學生與不同知識類型在這個時間窗口上呈現出差異化模式\[2]。

        為了方便讀者探索這些發現，我們在此建立一個互動式儀表板，利用模擬資料重現提案中的關鍵現象，
        並補充相關的科學知識，例如記憶鞏固理論、Bjork 的「適度困難」假說、間隔效應 (Spacing Effect) 等。
        """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("設定與說明")
    st.sidebar.markdown(
        """
        - **樣本數**：控制模擬資料的資料量。
        - **展示原理**：選擇要顯示的圖表與分析項目。
        - **模型分析**：訓練簡易隨機森林模型，並展示特徵重要性。
        """
    )

    # User controls
    n_samples = st.sidebar.slider("模擬樣本數", min_value=100, max_value=2000, step=100, value=600)
    show_u_curve = st.sidebar.checkbox("U 型鞏固曲線", value=True)
    show_response_time = st.sidebar.checkbox("反應時間分析", value=False)
    show_group_diff = st.sidebar.checkbox("能力/知識類型差異", value=False)
    show_model = st.sidebar.checkbox("隨機森林預測模型", value=False)

    # Generate synthetic data
    df = generate_synthetic_data(n_samples)

    st.markdown("## 合成資料概覽")
    st.write(
        "下方表格展示了模擬的學習者資料。`practice_lag` 表示學習後多久進行首次練習（小時），"
        "`accuracy` 為模擬的首次答對率，`response_time` 為反應時間。"
    )
    st.dataframe(df.head(10))

    # Plot U-shaped curve
    if show_u_curve:
        st.markdown("### U 型鞏固曲線")
        st.markdown(
            """
            下圖呈現練習時間延遲與平均答對率之間的關係。根據提案中的分析，6–8 小時
            為記憶鞏固的黃金期，我們可以看到在此範圍內準確率達到高峰\[1]。
            """
        )
        # Aggregate mean accuracy by lag bins
        lag_bins = pd.cut(df["practice_lag"], bins=np.arange(0, 25, 2))
        agg = df.groupby(lag_bins)["accuracy"].mean().reset_index()
        # Plot with matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        centers = [interval.mid for interval in agg["practice_lag"]]
        ax.plot(centers, agg["accuracy"], marker="o")
        # Use English axis labels to avoid missing glyphs
        ax.set_xlabel("Practice lag (hours)")
        ax.set_ylabel("Mean accuracy")
        ax.set_title("U‑shaped relationship between practice lag and accuracy")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Plot response time vs lag
    if show_response_time:
        st.markdown("### 認知負荷分析 (反應時間)")
        st.markdown(
            "隨著延遲時間增加，反應時間顯著上升，這與『適度困難』理論相符\[3]。"
        )
        # Aggregate median response time by lag bins
        lag_bins = pd.cut(df["practice_lag"], bins=np.arange(0, 25, 3))
        agg = df.groupby(lag_bins)["response_time"].median().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        centers2 = [interval.mid for interval in agg["practice_lag"]]
        ax2.plot(centers2, agg["response_time"], color="tab:orange", marker="s")
        ax2.set_xlabel("Practice lag (hours)")
        ax2.set_ylabel("Median response time (s)")
        ax2.set_title("Practice lag vs. response time")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

    # Plot group differences
    if show_group_diff:
        st.markdown("### 能力組與知識類型差異")
        st.markdown(
            "高分組學生在各延遲區間的表現較為穩定；潛力組學生對 6–8 小時區間格外敏感\[2]。\n"
            "同時，概念型知識更容易出現 U 型鞏固曲線\[3]。"
        )
        # Ability group plot
        fig3, ax3 = plt.subplots(1, 2, figsize=(12, 4))
        for ability_group, axis in zip(["High", "Low"], ax3):
            sub = df[df["ability"] == ability_group]
            bins = pd.cut(sub["practice_lag"], bins=np.arange(0, 25, 3))
            agg = sub.groupby(bins)["accuracy"].mean().reset_index()
            centers = [interval.mid for interval in agg["practice_lag"]]
            axis.plot(centers, agg["accuracy"], marker="o")
            # English axis labels
            axis.set_title(f"Ability group: {ability_group}")
            axis.set_xlabel("Practice lag (hours)")
            axis.set_ylabel("Mean accuracy")
            axis.set_ylim(0, 1)
            axis.grid(True, alpha=0.3)
        # For readability combine Medium into last panel
        fig_med, ax_med = plt.subplots(figsize=(6, 4))
        sub_med = df[df["ability"] == "Medium"]
        bins_med = pd.cut(sub_med["practice_lag"], bins=np.arange(0, 25, 3))
        agg_med = sub_med.groupby(bins_med)["accuracy"].mean().reset_index()
        centers_med = [interval.mid for interval in agg_med["practice_lag"]]
        ax_med.plot(centers_med, agg_med["accuracy"], marker="o", color="tab:gray")
        ax_med.set_title("Ability group: Medium")
        ax_med.set_xlabel("Practice lag (hours)")
        ax_med.set_ylabel("Mean accuracy")
        ax_med.set_ylim(0, 1)
        ax_med.grid(True, alpha=0.3)
        st.pyplot(fig3)
        st.pyplot(fig_med)

        # Knowledge type plot
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        for ktype, color in zip(["Fact", "Concept"], ["tab:green", "tab:purple"]):
            sub = df[df["knowledge_type"] == ktype]
            bins = pd.cut(sub["practice_lag"], bins=np.arange(0, 25, 3))
            agg = sub.groupby(bins)["accuracy"].mean().reset_index()
            centers = [interval.mid for interval in agg["practice_lag"]]
            ax4.plot(centers, agg["accuracy"], marker="o", color=color, label=ktype)
        ax4.set_xlabel("Practice lag (hours)")
        ax4.set_ylabel("Mean accuracy")
        ax4.set_title("Consolidation curves by knowledge type")
        ax4.legend(title="Knowledge type")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)

    # Model analysis
    if show_model:
        st.markdown("### 隨機森林預測模型")
        st.markdown(
            """
            為了驗證各因素對學習成效的預測能力，我們使用簡易的隨機森林分類器來預測學習者
            是否能在首次練習中答對。該模型的特徵包括練習延遲、學習者能力指標以及知識類型\[4]。
            後續可以根據真實資料進行調整。
            """
        )
        results = train_random_forest(df)
        st.write(f"模型準確率 (Accuracy)：{results['accuracy']:.3f}")
        st.write(f"F1 分數 (F1‑Score)：{results['f1']:.3f}")
        # Plot feature importance
        importances = results["importances"]
        features = list(importances.keys())
        values = list(importances.values())
        fig5, ax5 = plt.subplots(figsize=(6, 4))
        bars = ax5.barh(features, values, color="tab:blue")
        ax5.set_xlabel("Feature importance")
        ax5.set_title("Random forest feature importance analysis")
        ax5.invert_yaxis()
        for bar in bars:
            width = bar.get_width()
            ax5.text(width + 0.01, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center")
        st.pyplot(fig5)
        st.markdown(
            "從特徵重要性可以看出，練習延遲對預測學習成效的影響最大，"\
            "學習者能力次之，而知識類型為輔助因素，與提案中的假設一致\[5]。"
        )

    # Additional information section
    st.markdown("## 專業知識補充")
    st.markdown(
        """
        **記憶鞏固與間隔效應：**
        - 記憶鞏固是指新形成的記憶在時間中變得穩定的過程。研究表明，在初次學習後一段時間內，大腦會重新激活並強化相關神經網絡，使記憶從短期儲存轉移到長期儲存【432059056011258†L162-L172】。
        - 間隔效應（Spacing Effect）描述了將學習分散到多個時段比集中在一次學習更有效的現象。適當的間隔能夠引發輕微的遺忘，增加提取難度，反而有助於長期記憶。
        - "適度困難" 理論由 Bjork 等人提出，指出在學習過程中引入適當的挑戰，例如間隔練習、混合不同概念或以變化的情境復習，可以促使大腦更深層處理資訊，提升鞏固效果。

        **實務建議：**
        - 在數位學習平台中設計 **動態複習排程器**，根據學習者的表現與知識類型，智能推送複習任務到合適的時間點，避免過早或過晚。
        - 教師可參考 U 型鞏固曲線的規律，將複習安排在課程結束後 6–8 小時，尤其針對程度較弱的學生。
        - 對於記憶型知識與理解型知識，應採用不同的複習策略：記憶型知識可以更頻繁複習；理解型知識可以適度延遲以激發「適度困難」。
        """,
        unsafe_allow_html=True,
    )

    # Appendix: display original report figures
    st.markdown("## 報告圖表")
    st.markdown(
        "以下圖表直接來自報告，展示真實數據的視覺化結果，可與合成資料相互參照。"
    )
    # Helper to load images from disk as bytes. Reading the files here avoids Streamlit's
    # media file manager issues with relative file paths.
    def load_image_bytes(path: str):
        try:
            with open(path, "rb") as f:
                data = f.read()
            return data
        except Exception:
            return None

    # Load images as bytes
    img1 = load_image_bytes("images/fig1_avg_accuracy.png")
    img2 = load_image_bytes("images/fig2_feature_importance.png")
    img3 = load_image_bytes("images/fig3_ability_groups.png")
    img4 = load_image_bytes("images/fig4_knowledge_types.png")
    img5 = load_image_bytes("images/fig5_response_time_distribution.png")
    img6 = load_image_bytes("images/fig6_practice_lag_distribution.png")

    # Display images in pairs. Use use_container_width parameter instead of deprecated use_column_width.
    st.image(
        [img1, img2],
        caption=[
            "圖一：不同時間延遲區間的平均正確率",
            "圖二：預期之特徵重要性排序示意圖",
        ],
        use_container_width=True,
    )
    st.image(
        [img3, img4],
        caption=[
            "圖三：不同能力組學生的平均正確率隨時間延遲的變化",
            "圖四：不同認知技能的平均正確率隨時間延遲的變化",
        ],
        use_container_width=True,
    )
    st.image(
        [img5, img6],
        caption=[
            "圖五：不同時間延遲區間的答題分佈（反應時間）",
            "圖六：練習時間延遲分佈圖",
        ],
        use_container_width=True,
    )

    # Footnotes for citations used in the app.  Each number corresponds to
    # a line citation from the original proposal; the actual citation text is not
    # displayed here, but readers can refer back to the proposal for full context.
    st.markdown(
        """
        #### 參考註釋
        [1] 研究指出在學習後約 6–8 小時進行首次練習能顯著提升答對率【432059056011258†L162-L164】

        [2] 能力組差異：高分組表現穩定，潛力組對 6–8 小時區間特別敏感【432059056011258†L165-L167】

        [3] 知識類型差異：概念型知識更容易出現 U 型鞏固曲線【432059056011258†L169-L171】

        [4] 隨機森林模型特徵：包括練習延遲、學習者能力及知識類型【432059056011258†L121-L127】

        [5] 模型分析結論：練習延遲的影響最大，學習者能力次之，知識類型為輔助因素【432059056011258†L130-L154】
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()