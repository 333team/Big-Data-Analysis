
# 🐍 Python 專案協作 SOP

## ⚠️ 黃金準則 (Absolutely Important)
1.  **永遠不要** 直接在 `main` 分支上修改程式碼。
2.  **永遠不要** 使用 `Force Push` (強制推送)。
3.  **開始工作前**，一定要先更新 (`Pull`) 最新進度。
4.  **修改程式前**，一定要先建立新分支 (`Branch`)。

---

## 🛠️ 第一階段：環境建置 (Initial Setup)

請依據你的編輯器選擇對應的操作步驟。這部分只需做一次。

### 🔵 給 VS Code 使用者

**1. 安裝必要套件**
*   請確認已安裝 **Python** 擴充套件 (Microsoft 出品)。
*   推薦安裝 **Git Graph** (可以看到漂亮的分支圖)。

**2. 下載專案 (Clone)**
1.  打開 VS Code。
2.  按下 `F1` 或 `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`) 開啟指令列。
3.  輸入 `Git: Clone` 並按下 Enter。
4.  貼上專案的 GitHub 網址：`https://github.com/你的帳號/專案名稱.git`。
5.  選擇電腦中的一個資料夾存放，下載完成後點擊 **Open**。

**3. 建立虛擬環境 (Virtual Environment)**
1.  在 VS Code 中按下 `Ctrl + ~` (波浪號) 打開下方終端機 (Terminal)。
2.  輸入以下指令建立虛擬環境：
    *   **Windows**: `python -m venv .venv`
    *   **Mac/Linux**: `python3 -m venv .venv`
3.  **重要步驟**：VS Code 右下角會跳出通知 *"We noticed a new virtual environment has been created. Do you want to select it for the workspace folder?"*
    *   請務必點選 **Yes**。
    *   *如果沒跳出*：按 `F1` -> 輸入 `Python: Select Interpreter` -> 選擇有 `('.venv': venv)` 字樣的選項。

**4. 安裝依賴套件**
1.  確認終端機最前方有出現 `(.venv)` 字樣 (代表已進入虛擬環境)。
    *   *若無*：關掉終端機 (`垃圾桶圖示`) 再開一個新的就會出現。
2.  輸入指令：`pip install -r requirements.txt`。

---

### 🟢 給 PyCharm 使用者

**1. 下載專案 (Clone)**
1.  打開 PyCharm，在歡迎畫面點擊 **Get from VCS** (或右上角 `Git` -> `Clone`)。
2.  **URL** 欄位貼上：`https://github.com/你的帳號/專案名稱.git`。
3.  **Directory** 選擇存放位置，點擊 **Clone**。

**2. 設定虛擬環境 (Virtual Environment)**
PyCharm 通常會自動偵測，但為了保險請檢查：
1.  點擊右下角的 `<No Interpreter>` 或目前的 Python 版本號文字。
2.  選擇 **Add New Interpreter** -> **Add Local Interpreter**。
3.  選擇 **Virtualenv Environment**。
    *   **Location**: 確保路徑結尾是 `/venv` 或 `/.venv` (專案資料夾內)。
    *   **Base interpreter**: 選擇你電腦安裝的 Python。
    *   點擊 **OK**。
4.  PyCharm 會開始建立環境 (右下角會有進度條)，請耐心等待。

**3. 安裝依賴套件**
1.  打開專案中的 `requirements.txt` 檔案。
2.  PyCharm 頂部通常會跳出黃色通知 *"Package requirements are not satisfied"*。
3.  點擊 **Install requirements** 即可。
    *   *若沒跳出*：點擊下方 **Terminal** 標籤，輸入 `pip install -r requirements.txt`。

---

## 🔄 第二階段：日常開發流程 (Daily Workflow)

這是你每天要重複做的動作。請養成肌肉記憶。

### 🔵 VS Code 操作流程

#### 步驟 1：同步最新進度 (Sync)
1.  看左下角狀態列，確認目前分支是 `main`。
    *   如果不是，點擊分支名稱 -> 選擇 `main`。
2.  點擊左側選單的 **原始碼控制 (Source Control)** (像樹枝的圖示)。
3.  點擊選單右上角的 `...` -> **Pull** (拉取)。
    *   *或者*：點擊左下角的同步圖示 (圓圈箭頭)。

#### 步驟 2：建立新分支 (Branch)
1.  點擊左下角的 `main` 字樣。
2.  在上方選單選擇 **+ Create new branch...** (建立新分支)。
3.  **輸入分支名稱** (範例：`feature/login-page` 或 `fix/data-error`)。
4.  按下 Enter，左下角現在應顯示你的新分支名稱。

#### 步驟 3：寫程式與提交 (Commit)
1.  修改你的程式碼。
2.  進入 **原始碼控制 (Source Control)** 面板。
3.  你会看到 "Changes" 下列出你改過的檔案。
4.  點擊檔案旁邊的 **+** 號 (Stage Changes)，將檔案移到 "Staged Changes" 區塊。
5.  在上方 "Message" 欄位輸入修改說明 (例如：「新增登入按鈕樣式」)。
6.  點擊藍色的 **Commit** 按鈕。

#### 步驟 4：推送到雲端 (Push)
1.  點擊藍色的 **Publish Branch** (發佈分支) 按鈕。
2.  之後若有修改，按鈕會變成 **Sync Changes**，點擊即可推送。

---

### 🟢 PyCharm 操作流程

#### 步驟 1：同步最新進度 (Sync)
1.  看視窗**右下角**或**頂部 Git 工具列**，確認目前分支顯示 `main`。
    *   如果不是，點擊它 -> 選擇 `main` -> **Checkout**。
2.  點擊頂部導航列的藍色箭頭圖示 (Update Project)。
    *   或者：右鍵點擊專案資料夾 -> **Git** -> **Pull**。
3.  選擇 **Merge** 或 **Rebase** (預設即可)，點擊 OK。

#### 步驟 2：建立新分支 (Branch)
1.  點擊**右下角** (或頂部) 的 `main` 分支名稱。
2.  選擇 **+ New Branch**。
3.  **Name** 輸入分支名稱 (範例：`feature/login-page`)。
4.  勾選 **Checkout branch**。
5.  點擊 **Create**。

#### 步驟 3：寫程式與提交 (Commit)
1.  修改你的程式碼。
2.  點擊左側選單的 **Commit** 標籤 (或按下 `Ctrl+K` / `Cmd+K`)。
3.  在 Changes 清單中，**勾選**你要提交的檔案。
4.  在下方 Commit Message 區域輸入說明。
5.  點擊右下角的 **Commit** 按鈕。

#### 步驟 4：推送到雲端 (Push)
1.  點擊頂部的 **綠色箭頭** (Push) 圖示 (或按下 `Ctrl+Shift+K` / `Cmd+Shift+K`)。
2.  確認此時是要推送到你的新分支。
3.  點擊 **Push**。

---

## 🚀 第三階段：合併程式碼 (Pull Request)

無論你用哪個編輯器，這一步驟統一在 **GitHub 網頁** 上進行。

1.  **發起請求**：
    *   當你 Push 完後，打開 GitHub 專案首頁。
    *   你會看到一個黃色通知框 *"Compare & pull request"*，點擊它。
    *   (若沒看到，去 "Pull requests" 分頁點 "New pull request" -> 選擇你的分支)。

2.  **填寫資訊**：
    *   **Title**: 簡單清楚地說明你做了什麼。
    *   **Description**: 描述細節、測試方式。
    *   **Reviewers** (右側欄位): **務必選取組長 (你的名字)**。

3.  **送出**：
    *   點擊綠色的 **Create pull request**。

4.  **等待審核**：
    *   通知組長：「我發 PR 了，麻煩看一下」。
    *   如果組長要求修改 (Changes requested)，請在本地修改程式 -> 存檔 -> Commit -> Push。
    *   GitHub 上的 PR 會自動更新你的修改，**不需要**重新開一個 PR。

5.  **合併完成**：
    *   當組長核准並合併後，你的工作就完成了。
    *   回到編輯器，切換回 `main` 分支，執行 **Pull**，準備下一個工作。

---

## 🆘 常見災難救援 (FAQ)

**Q1: 我忘記切換分支，不小心在 `main` 上面改了程式怎麼辦？**
*   **VS Code**:
    1.  不要驚慌，還不要 Commit。
    2.  直接點擊左下角建立新分支 (`feature/xxx`)。
    3.  你的修改會自動帶過去新分支，現在可以安全 Commit 了。
*   **PyCharm**:
    1.  同上，直接 New Branch，修改會自動帶過去 (Smart Checkout)。

**Q2: Push 的時候被拒絕 (Rejected)？**
*   這通常是因為別人在你之前改了同一個分支 (或你是多人共用分支)。
*   執行 **Pull** (拉取) -> 解決衝突 (Conflict) -> 再次 Push。

**Q3: 遇到「衝突 (Conflict)」怎麼辦？**
*   編輯器會標示出紅色的區域。
*   `<<<<<<<` 是你的程式，`>>>>>>>` 是傳入的程式。
*   **VS Code**: 上方會有按鈕 "Accept Current" (保留你的) 或 "Accept Incoming" (保留對方的)。選完後存檔提交。
*   **PyCharm**: 會跳出一個三欄視窗，左邊是你，右邊是對方，中間是結果。點擊 `>>` 或 `<<` 來決定要用誰的，最後按 Apply。