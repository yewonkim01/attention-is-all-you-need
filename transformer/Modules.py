import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature  #논문에 있는 d_k인 것 같음: 내적값이 커지면 -> softmax 기울기가 0인 영역에 도달할 가능성이 높음(S자)-> 역전파할 때 기울기 작아져서 학습 불안 -> √dk로 스케일링(softmax 기울기가 0인 영역까지 도달하지 못하도록)
        self.dropout = nn.Dropout(attn_dropout) #특정 노드들을 deactivate하는 기능

    def forward(self, q, k, v, mask=None):
        #가중치 == score 같은 말인지???
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) #q*k.T #3번째, 4번째 차원을 transpose 근데 왜 3,4번째 차원??

        """
        이 attention은 여러 곳에서 사용되지만
        **decoder**에서는 뒷단어를 cheating하지 못하도록 해줘야하기 때문에
        mask 변수로 처리해주어야함
        """
        if mask is not None:
            #decoder에서 i번째 position을 예측할 때는 i-1번까지의 토큰들만 예측에 이용해야함(아니면 cheating임)
            #tensor.masked_fill() : torch 함수) tensor의 특정값을 다른 값으로 바꾸고자할 때 사용
            attn = attn.masked_fill(mask == 0, -1e9) #mask==0이면 음의 무한대로 바꾸어줌 -> softmax 지나면 0이 되어서 반영 안됨
        attn = self.dropout(F.softmax(attn, dim=-1)) #가장 마지막 차원에 대해서 softmax => 마지막 차원이 뭔지 모르겠음
        output = torch.matmul(attn, v) #지금까지 구한 attention score를 실제 값인 value에 곱해줌

        return output, attn  #출력값이랑 attention 가중치 넘겨줌
    


