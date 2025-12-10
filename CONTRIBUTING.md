# 🐍 Python 團隊專案開發手冊 (SOP)

歡迎加入本專案！由於這是一個 **Private (私人)** 專案，且我們採用嚴格的 Git Flow 版本控制，請務必按照以下步驟進行設定與開發。

## ⚠️ 黃金準則 (絕對要遵守)
1.  **接受邀請**：沒接受邀請前，你連專案都看不到。
2.  **嚴禁直推 Main**：永遠不要直接在 `main` 分支修改程式碼。
3.  **先分支，再開發**：修改任何程式碼前，一定要先建立新分支。
4.  **保持同步**：開工前，一定要先 Pull 更新進度。

---

## 🏁 第一階段：加入專案 (Access)

因為這是私人專案，你無法直接 Clone。

1.  **收取邀請信**：
    *   前往你的 Email 信箱，尋找標題包含 `[GitHub] ... invited you to join` 的信件。
    *   點擊信中的 **View invitation** -> **Accept invitation**。
    *   *或者*：直接登入 GitHub 網站，點擊右上角鈴鐺 🔔，在通知中按 **Accept**。
2.  **確認權限**：
    *   點擊專案網址，確認你能看到程式碼列表，而不是 404 錯誤。

---

## 🛠️ 第二階段：環境建置 (Setup)

請依據你使用的編輯器選擇對應步驟。

### 🔵 選項 A：使用 VS Code

**1. 準備工具**
*   確認已安裝 **VS Code** 與 **Python**。
*   **強烈推薦**：安裝 **Git Graph** 擴充套件 (方便看線圖)。

**2. 下載專案 (Clone)**
1.  打開 VS Code，按下 `F1` 或 `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`)。
2.  輸入 `Git: Clone` 並 Enter。
3.  貼上專案網址：`https://github.com/你的帳號/專案名稱.git`。
4.  **登入授權**：
    *   系統會跳出 *"Sign in to GitHub..."* 提示。
    *   點擊 **Allow** -> 瀏覽器彈出授權頁 -> 點擊 **Authorize** -> 允許開啟 VS Code。
5.  選擇存放資料夾 -> 下載完成後點擊 **Open**。

**3. 建立虛擬環境**
1.  按下 `Ctrl + ~` 開啟終端機。
2.  輸入：
    *   Windows: `python -m venv .venv`
    *   Mac: `python3 -m venv .venv`
3.  **關鍵一步**：右下角跳出 *"We noticed a new virtual environment..."* 時，務必點選 **Yes**。

**4. 安裝套件**
1.  確認終端機前方有 `(.venv)` 字樣。
2.  輸入：`pip install -r requirements.txt`。

---

### 🟢 選項 B：使用 PyCharm

**1. 綁定帳號 (Auth)**
1.  打開 PyCharm -> **Settings** (Windows: `Ctrl+Alt+S` / Mac: `Cmd+,`)。
2.  點選 **Version Control** -> **GitHub**。
3.  點擊 `+` 號 -> **Log In via GitHub** -> 在瀏覽器點擊 **Authorize JetBrains**。

**2. 下載專案 (Clone)**
1.  在歡迎畫面點擊 **Get from VCS** (或主畫面右上角 `Git` -> `Clone`)。
2.  左側選單點 **GitHub**，你應該會看到此專案在列表中 (因為你已接受邀請)。
3.  選中專案，點擊 **Clone**。

**3. 設定虛擬環境**
1.  PyCharm 通常會自動偵測。若出現 *"No Python interpreter configured"*：
2.  點擊右下角 `<No Interpreter>` -> **Add New Interpreter** -> **Add Local Interpreter**。
3.  選擇 **Virtualenv** -> 確保路徑在專案內的 `.venv` -> OK。

**4. 安裝套件**
1.  打開 `requirements.txt` 檔案。
2.  點擊頂部黃色警告條的 **Install requirements**。

---

## 🔄 第三階段：日常開發循環 (Daily Workflow)

這是你每天要重複做的動作，請養成肌肉記憶。

### 🔵 VS Code 操作流程

#### 1. 同步 (Sync)
*   **目的**：確保你的 `main` 是最新的，避免衝突。
*   **動作**：
    1.  點擊左下角分支名稱，切換到 `main`。
    2.  點擊左側 **原始碼控制 (Source Control)** 圖示。
    3.  點擊左下角的 **同步圖示 (🔄)** 或選單中的 **Pull**。

#### 2. 分支 (Branch)
*   **目的**：隔離你的開發環境。
*   **動作**：
    1.  點擊左下角 `main`。
    2.  選單上方點 **+ Create new branch**。
    3.  **命名規則**：
        *   新功能：`feature/你的功能名` (例 `feature/login-ui`)
        *   修 Bug：`fix/錯誤名稱` (例 `fix/api-timeout`)
    4.  輸入完按 Enter。

#### 3. 提交 (Commit)
*   **動作**：
    1.  寫程式、存檔。
    2.  點擊 **原始碼控制**。
    3.  按檔案旁的 `+` (Stage Changes)。
    4.  輸入清楚的訊息 (例：「完成登入頁面切版」)。
    5.  按 **Commit**。

#### 4. 推送 (Push)
*   **動作**：
    1.  按藍色的 **Publish Branch**。
    2.  (若之後有修改) 按 **Sync Changes**。

---

### 🟢 PyCharm 操作流程

#### 1. 同步 (Sync)
*   **動作**：
    1.  確認右下角顯示 `main`。
    2.  點擊上方導航列的 **藍色箭頭 (Update Project)**。
    3.  選擇 **Merge** -> OK。

#### 2. 分支 (Branch)
*   **動作**：
    1.  點擊右下角 `main`。
    2.  選擇 **+ New Branch**。
    3.  輸入名稱 (例 `feature/login-ui`) -> 勾選 Checkout -> Create。

#### 3. 提交 (Commit)
*   **動作**：
    1.  點擊左側 **Commit** 標籤 (`Alt+0` 或 `Cmd+0`)。
    2.  勾選要提交的檔案。
    3.  輸入訊息 -> 點擊右下角 **Commit**。

#### 4. 推送 (Push)
*   **動作**：
    1.  點擊上方 **綠色箭頭 (Push)** (`Ctrl+Shift+K`)。
    2.  確認分支無誤 -> **Push**。

---

## 🚀 第四階段：發起合併請求 (Pull Request)

無論你用哪個編輯器，這一步請回到 **GitHub 網頁** 操作。

1.  **發起 PR**：
    *   Push 後，GitHub 首頁會出現黃色框框 *"Compare & pull request"*，點擊它。
2.  **填寫資訊**：
    *   **Title**：簡述做了什麼。
    *   **Reviewers** (右側欄位)：**一定要選組長**。
3.  **建立**：
    *   點擊 **Create pull request**。
4.  **修改 (若被退回)**：
    *   若組長要求修改，**不用**關閉 PR。
    *   在本地電腦修改 -> Commit -> Push。
    *   PR 會自動更新，並通知組長再次檢查。
5.  **合併**：
    *   組長核准並合併後，你的任務完成。
    *   **刪除分支**：GitHub 上會提示 Delete branch，可以點擊刪除。
    *   回到電腦，切換回 `main`，Pull 最新進度，準備下一個任務。

---

## 🆘 常見問題急救包 (FAQ)

**Q1: Push 的時候出現 `Authentication failed` 或一直問密碼？**
*   **原因**：GitHub 不再支援密碼登入，需用 Token 或 OAuth。
*   **解法**：
    *   VS Code: 點左下角人頭 -> Sign out -> 再按 Sync 觸發重新登入。
    *   PyCharm: Settings -> Version Control -> GitHub -> 刪除舊帳號重新 Log In。

**Q2: 忘記開分支，直接在 `main` 改了程式？**
*   **千萬不要 Commit**。
*   直接建立新分支 (Create Branch)。
*   你的修改會自動「帶過去」新分支，這時再 Commit 即可。

**Q3: 遇到 Conflict (衝突) 怎麼辦？**
*   這代表別人在你修改的同一行程式碼做了變更。
*   **VS Code**: 選擇 `Accept Incoming` (用對方的) 或 `Accept Current` (用你的)，存檔後提交。
*   **PyCharm**: 會跳出三欄視窗，中間是結果。點擊 `<<` 或 `>>` 決定保留哪邊，最後按 Apply。