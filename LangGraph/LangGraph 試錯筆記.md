# 2024.12.04

## 發想
有了multi Agent的概念之後，開始想拿各種東西進行串接，  
之前的詞根詞綴研究發現內容太廣，現在的水平還做不太出來，  
所以打算改先拿簡單的內容開始研究，  
timer!!  
我決定先讓chat bot可以透過對話增加計時器功能，  
在嘗試的路上遇到了很多的問題，並且一步步解決    
首先開始建立了一個簡易的timer，在外部運行良好  

## Try1. UI更改  
第一個需要解決的問題是拔掉按鍵，並使用參數控制   
簡化的版本只吃一個參數，計時時間,這部分的更改沒有花多少時間  

## Try 2. Agent懂不懂使用tool_call參數   
爬了不少文，有些建議把參數統一放在TypedDict 進行管理而不要使用tool_call 的參數    
但我還是覺得使用tool_call 比較符合我的想像    
但我不確定Agent 是否知道如何使用tool_call 的參數    
嘗試了多次之後發現可以，但在註解裡面必須要定義非常明確。  
反而花了比較多的時間在調整註解  

## Try 3. Timer功能應該增加線程  
從設計理念來看，當Timer開始對話  
由於Agent沒有收到tool_call 的return  
對話就會卡在那裡無法繼續，直至計時器結束  
所以線程的設計發現很有必要  
第一次包線程包錯了地方   
我沒有把創建TK 跟mainloop一起包進去  
所以導致線程沒有達到預期效果  

## Try 4. 是否應該把 timer 包成一個完整的app?
原本享用subprocess實現這一點，  
將全部的工作都開獨立線程在邏輯上似乎有點微妙  
改成使用外部APP似乎比較適合一點  
但使用subprocess後，  
Agent 又開始等回應了  
原因明顯是thread的包法有問題，但因為我比較想看的是多Agent 的玩法  
subprocess的概念先暫時擱置  

## 小結
至此為止完成了以下程式  
[AI Agent 計時器 - v1](https://github.com/shwpatrick/Study-Note/blob/main/LangGraph/AI%20Agent%20%E8%A8%88%E6%99%82%E5%99%A8%20-%20v1.%E5%8F%AA%E6%9C%89%E8%A8%88%E6%99%82%E5%99%A8%E4%B8%80%E5%80%8Bchatbot.ipynb)  
裡面有不少試錯用的block 有空再來整理註解  
