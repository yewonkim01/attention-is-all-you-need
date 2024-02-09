''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"


"""
attention head를 여러 개 두면, 모델이 동시에 여러 측면에 초점을 맞출 수 있다
ex) 한 헤드는 주어-동사 관계에, 다른 한 헤드는 인접한 형용사가 있는지 등에 초점을 맞추어
여러개의 head를 두면 문장 내 여러 관계를 파악할 수 있음 (like CNN의 필터수)
"""
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head  #몇개의 head를 쓸지(몇개의 측면에 초점을 맞출지?)
        self.d_k = d_k        #d_k는 query와 key의 dimension : d_model(단어 임베딩의 차원) / h(헤드수)
        self.d_v = d_v        #d_k == d_v??

        
        #n_head * d_k == d_model?
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False) #쿼리를 만드는 가중치
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)  #키를 만드는 가중치
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False) #value를 만드는 가중치
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5) #scaling 적용해서 dot-product

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        #q.size(0)은 행 개수
        #q.size(1)은 열 개수
        
        #len_* : *의 시퀀스 길이
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv

        #view는 Numpy의 reshape과 같이 차원 변경
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        # doct product를 위해서 tranpose??
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            #mask있으면 unsqueeze로 차원추가해서 mask처리
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        #attention 결과
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        #contiguous: 주소값을 연속적으로 만들어줌
        """
        contiguous()가 필요한 이유는, sequence의 순서가 중요하기 때문에
        이를 보장해주기 위해서라고함
        """
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual  

        q = self.layer_norm(q)

        return q, attn

#self-attention을 거친 후에 통과
#position마다(==각 개별 단어마다) 적용되기 때문에 "position-wise" (fully connected임)
# 선형변환 -> ReLU -> 선형변환 으로 이루어짐 FFN(x) = max(0, xW1 + b1)W2 + b2
    ### 왜하는지 모르겠음?
class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise   #원래 들어온 벡터의 차원과 맞춰준다?
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6) #각 input을 평균 0, 분산 1로 정규화
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x  #처리하지 않은 원본 텐서를 다음층으로 "그대로" 전달

        x = self.w_2(F.relu(self.w_1(x)))  #position-wise feed forward 계산
        x = self.dropout(x) #특정 노드들 deactivate -> 일부 뉴런에 의존하는 것 방지-> 과적합 방지
        x += residual #처리된 텐서에 더해줌 -> 근데 원본 값에 + 처리된 값이면 값이 중복되는 거 아닌가?
                      #학습이 용이해진다고 함
        
        #왜 relu 사용??

        x = self.layer_norm(x) 

        return x
