# 概覽

[資料與圖源](#資料與圖源)
[空間架構(SA)概念](#空間架構(SA)概念)  
[CNN層的運算需求](#CNN層的運算需求)  
[CNN的問題考量與解決](#CNN的問題考量與解決)  
[實際資料流的策略設計考量](#實際資料流的策略設計考量)  
[列固定資料流實作概念](#列固定資料流實作概念)  
[列固定資料流的硬體實務](#列固定資料流的硬體實務)  
[列固定資料流與CNN各層的交互](#列固定資料流與CNN各層的交互)  
[Eyeriss架構對於資料流傳輸的特化設計](#Eyeriss架構對於資料流傳輸的特化設計)  
[資料傳輸引起延遲與處理](#資料傳輸引起延遲與處理)

# 資料與圖源

[Paper : Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks](https://eems.mit.edu/wp-content/uploads/2016/04/eyeriss_isca_2016.pdf)  

# 空間架構(SA)概念

AI有大量的運算，所以透過較佳的資料搬運與儲存策略可以減少大量成本  
解決問題 ： 盡可能降低運算時資料搬運的成本  

空間架構可以以此理解分層，越低層級間的交互就可以越少耗能  
所以設計目標就是盡可能在低層級完成運算與傳輸  


這個系統提供了四個資料存取的儲存階層

| 層級高到低(越低能耗越小)                                                              |                         |
| -------------------------------------------------------------------------- | ----------------------- |
| DRAM                                                                       | 最外圍、最貴                  |
| Global Buffer                                                              | 常用於DRAM資料緩存, 權重、計算中間結果等 |
| 陣列內部（PE 間通訊）                                                       | PE與PE間的資料搬運，空間架構影響了對傳距離 |
| RF（本地暫存） |                         |

關鍵字：  
空間架構(SA)：PE之間可以互傳，所以PE陣列的組成方式很大影響傳輸距離  
數據流(Data flow)：PE與PE間的資料搬運策略  
PE : Processing Engine
通常各個平台有各自的數據流最佳化策略  


# CNN層的運算需求

CNN 每層對於這些運算的實際需求與考量不同  

| 層 | 是否重要           |
| ---- | -------------- |
| CONV | 考量重點           |
| FC   | 考量重點           |
| POOL | 與CONV使用相同的解決方案 |
| ACT  | 可忽略            |

# CNN的問題考量與解決

## 資料處理問題

- 輸入資料存取問題：若直接從 DRAM 為所有 MAC 運算讀取輸入資料，會導致極高的頻寬需求與能耗。  
	- 資料重用（data reuse）方式來緩解  
 		- 卷積重用（convolutional reuse）：CONV層單獨  
   		- 濾波器重用（filter reuse）：CONV, FC層  
     		- ifmap重用（ifmap reuse）：CONV, FC層  
- 中間資料儲存壓力問題：平行 MAC 運算同時產生大量「部分和」（partial sums, psums），若無法立即進行累加，就需要額外的儲存空間與記憶體 R/W 能量。  
	- 使用 操作排程（operation scheduling）來處理
 	- 無法同時最大化輸入重用與立即累加 psum，因為來自相同濾波器或 ifmap 價值的 psum 無法直接合併。  
  	- 為達成高吞吐量與高能效，CNN 資料流必須同時考慮輸入資料重用與 psum 累加的排程策略  

## 自適應處理

CONV／FC 層可能擁有多種不同的結構 -> 硬體架構不能只支援特定形狀，而應支援動態映射至高效資料流的能力

## CNN 與 ISP 的不同卷積不同

雖然ISP也有考慮卷積優化，無法直接套用於 CNN 的處理中，因為  

- CNN CONV 是 4D (ISP 是 2D)  
- CNN 的Filter 不固定 (訓練計算得來)  
- ISP 設計時沒有考慮psum的問題

# 實際資料流的策略設計考量

- psum 計算的中間產物，可以暫不處理，但會帶來記憶體空間的壓力
- 重用性：許多架構可以重複使用，不需要每次重新讀取，減緩高帶寬需求和高能源消耗
	- convolutional reuse : CONV 層獨有
	- filter reuse : CONV, FC 層都有
	- ifmap reuse: CONV, FC 層都有
- Mapping: 每層的大小不同，所以需要考慮到實際運算元跟層的映射關係


| 資料流策略     | 核心概念                                     |
| --------- | ---------------------------------------- |
| 權重固定 WS   | RF固定放置權重                                 |
| 輸出固定 OS   | RF固定放置psum                               |
| 無本地重用 NLS | 大幅減少RF，把空間留給global buffer<br>重用在PE間交互中發生 |
| 列固定 RS    | 將高維卷積拆解為一維卷積                             |


- 權重固定（Weight Stationary, WS）資料流
	- 每個RF固定一個權重
	- RxR個權重映射給RxR個PE
	- 硬體使用：最大化重用權重 -> psum 基本全放在global buffer
	- buffer 的大小直接決定一次可以載多少個filter
- 輸出固定（Output Stationary, OS）資料流
	- RF固定為psum儲存空間
	- 每個輸出特徵圖 (ofmap) 像素的累加靜止在一個處理單元 (PE)
	- 為了選擇要處理的 4D 輸出特徵圖區域，OS 資料流有兩個主要的選擇1：
		- 選擇要處理多個輸出通道 (Multiple ofmap channels, MOC) 還是 單一輸出通道 (Single ofmap channel, SOC)1。
		- 選擇要處理多個輸出平面像素 (Multiple ofmap-plane pixels, MOP) 還是 單一輸出平面像素 (Single ofmap-plane pixel, SOP)1。
	- 多個輸出通道 vs 單一輸出通道
		- 多個輸出通道 (MOC)：表示 PE 陣列被用來同時處理屬於多個不同輸出通道的資料。
		- 單一輸出通道 (SOC)：表示 PE 陣列一次只處理屬於一個單一輸出通道的資料。
	- 結合這兩個選擇，產生了三種實際應用的 OS 資料流子類別
		- SOC-MOP：主要用於卷積層 (CONV layers)。它側重於一次處理單一輸出特徵圖平面
			- 還最大化利用 PE 陣列中的卷積資料再利用 (convolutional reuse
		- MOC-MOP：一次處理多個輸出特徵圖平面以及同一個平面中的多個像素
			- 試圖進一步利用卷積再利用和輸入特徵圖 (ifmap) 再利用
		- MOC-SOP：主要用於全連接層 (FC layers)3。它處理多個輸出通道，但每個通道一次只處理一個像素
			- 它側重於進一步利用輸入特徵圖再利用 (ifmap reuse)
	- **硬體使用**：所有 OS 資料流都將 RF 作為 psum 儲存空間，以實現固定累加。此外，SOC-MOP 與 MOC-MOP 為了實現卷積重用，還需額外的 RF 儲存空間以緩衝 ifmap
- 無本地重用（No Local Reuse, NLR）資料流
	- 不在 RF（暫存器檔案）層級進行資料重用
	- 使用 PE 間的通訊進行 ifmap 重用與 psum（部分和）累加
	- NLR 將 PE 陣列分成多個 PE 群組。同一群組內的 PE 會讀取相同的 ifmap 像素，但分別使用來自相同輸入通道的不同濾波器權重；而不同群組則讀取來自不同輸入通道的 ifmap 像素與濾波器權重。所產生的 psum 會在整個陣列中跨群組進行累加。
	- **硬體使用**：NLR 資料流不需要 RF 儲存空間。由於 PE 陣列僅由 ALU 計算單元構成，因此可以保留更多晶片面積給全域緩衝區（global buffer），用來儲存 psum 與可重用的輸入資料。
	- **範例**：NLR 資料流的變體實作了一種特殊暫存器，放置於每個 PE 陣列欄的末端，以保存 psum，從而**減少 psum 存取全域緩衝區的次數**。
- 節能資料流：列固定（Row Stationary, RS）
	- 問題：舊的策略不是最大化資料重用，就是最小化psum，但沒辦法兼顧
	- 概念發想：strip mining -> 將高維卷積拆成平行的一維卷積原語（1D convolution primitives）**
	- 每個 1D 原語會映射到一個處理單元（PE）上執行；因此，**每對輸入列與權重列的運算會固定在同一個 PE 中進行**，這就實現了在 **RF 層級的權重與 ifmap 像素的卷積重用**
	- 核心難點：如何將這些原語有效率地映射到 PE 陣列上

# 列固定資料流實作概念

![image](https://github.com/user-attachments/assets/739cb939-6f6e-4d94-92da-8aab6ba21140)


- 採用了**兩階段映射方式**：
	1. **邏輯映射（Logical Mapping）**
		- 將每個 **1D 卷積原語**對應到一個**邏輯 PE（logical PE）**上
		- 每個濾波器列與 ifmap 列分別在水平方向與對角線方向上被重用
		- 每行 psum 則沿垂直方向累加
	2. **實體映射（Physical Mapping）**：一階段折疊
		- **關鍵動作是「折疊（Folding）」**，即將多個邏輯 PE 的運算序列地安排在同一個實體 PE 上執行
		- **保留邏輯 set 內部的資料重用**（例如濾波器與 ifmap 的卷積重用）與 psum 的空間累加（依賴 PE 間通訊）
		- 跨多個 set 的資料重用機會也能被有效利用
		- 透過**將不同邏輯 set 中相同位置的邏輯 PE 映射至同一個實體 PE**，可以實現**RF 層級的資料重用**
	3. **實體映射（Physical Mapping）**：二階段折疊（Processing Pass Folding）
		- 第一階段折疊後，PE 陣列可以處理多個邏輯 PE set，稱為一個 **處理批次（processing pass）**
		- 一個處理批次**通常無法處理整個 CONV 層的所有 PE set**
		- 此階段中，**全域緩衝區（global buffer）用來儲存跨批次重用的輸入資料與 psum**
		
# 列固定資料流的硬體實務

- **RF（暫存器檔案）**：經過第一階段折疊後，在一個 PE 中執行多個 1D 卷積原語時，RF 可用來實現各種資料重用：
    
    - 單一原語內部的 **卷積重用（convolutional reuse）**        
    - 不同原語之間的 **濾波器重用（filter reuse）**與**ifmap 重用**
    - 原語內與原語間的 **psum 累加**
        
- **陣列層級（PE 間通訊）**：
    
    - 邏輯 PE set 內的卷積重用在這一層會被完全利用
    - 多個 set 空間映射到不同的實體 PE 陣列後，可進一步實現 filter 與 ifmap 重用
    - psum 的累加亦可在 set 內部與跨 set 間完成
        
- **全域緩衝區（Global Buffer）**：
    
    - 根據其容量，在第二階段折疊後，用來完成 RF 與陣列層級未處理完的：
        - **filter 重用**
        - **ifmap 重用**
        - **psum 累加**

# 列固定資料流與CNN各層的交互

RS 資料流是為了解決 CONV 層中的高維度卷積所設計，它也**能自然支援其他兩種類型的網路層**

- **全連接層（FC Layer）**：
    
    - FC 層的計算形式與 CONV 層相同，只是沒有卷積重用特性。
    - 由於 RS 資料流設計可涵蓋所有類型的資料移動，因此在 FC 層中仍可套用：
        - **filter 重用**
        - **ifmap 重用**
        - **psum 累加**
    - 無需像 OS 資料流一樣，在 SOC-MOP 與 MOC-SOP 間切換。
        
- **池化層（POOL Layer）**：
    
    - 將每個 PE 中的 MAC（乘加運算）替換為 MAX（最大值比較），即可處理池化層。
    - 假設 N=M=C=1N = M = C = 1N=M=C=1，分別處理每一個 fmap 平面。

#  Eyeriss架構對於資料流傳輸的特化設計

資料流是透過**三個不同的 NoC（網路通訊結構）**來處理三種不同資料：  
- 全域廣播 NoC（Global multicast NoC）：用於 ifmap 與濾波器的傳送  
- 本地 PE 對 PE NoC（Local PE-to-PE NoC）：用於 psum 的傳遞與累加

![image](https://github.com/user-attachments/assets/fdb677bc-5a8f-484a-8a46-dd7fc6456ce9)

進一步節能：活用稀疏性（sparsity）-> CNN經過剪枝（pruning）或稀疏訓練後，很多權重跟輸出的值都是0 

- 僅對非零值進行資料讀取與 MAC 運算
- 對資料進行壓縮，以減少資料傳輸成本

# 資料傳輸引起延遲與處理

資料傳輸也會引起延遲（特別是儲存頻寬受限時），但這類影響可透過以下常見技術緩解：

- 預取（Prefetching）
- 雙緩衝（Double Buffering）
- 快取（Caching）
- 管線化（Pipelining）

這些技術在 CNN 加速領域中已被證實能有效隱藏資料傳輸延遲  
模擬中預設資料移動對吞吐量影響不大  



