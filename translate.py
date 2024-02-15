''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants

#torchtext: pytorch 모델에 입력을 주는 텍스트 데이터셋을 구성하기 편하게 만들어주는 data loader(데이터셋을 샘플에 쉽게 접근할 수 있도록 iterable하게 감쌈
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator


def load_model(opt, device):

    #모델 정보 가져오기(모델 weight들이나 epoch .. 등등?)
    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    """Transformer:
    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            scale_emb_or_prj='prj'):
    """

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,   #target 언어의 embedding 레이어와 최종  linear proj 레이어(decoder의 마지막 layer) 간 가중치 공유 여부
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,  #source언어와 target 언어의 embedding할 때 가중치를 공유할지 여부
        d_model=model_opt.d_model,       #encoder-decoder hidden state의 차원수
        d_word_vec=model_opt.d_word_vec, #한 단어가 벡터로 표현될 때의 차원수
        d_inner=model_opt.d_inner_hid,   #d_inner: 내부 인코더 디코더 레이어 차원수
        n_layers=model_opt.n_layers,     #encoder /decoder 모두 N = 6 개의 동일한 layer가 stack되어있음
        n_head=model_opt.n_head,         #attention head의 수
        d_k=model_opt.d_k,   #64   word vector의 차원이 512인데 n_head = 8이므로 나눠서 처리 d_k = 64 == d_v
        d_v=model_opt.d_v,   #64
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''
    #argparse: 인자를 파싱(주어진 데이터를 해석하고 원하는 형식으로 변환)할 때 사용하는 라이브러리
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    #-data_pkl은 데이터 pickle파일: source의 vocab와 target의 vocab이 존재
    #eng -> kor 이면 source에는 eng voccab / target에는 kor vocab
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    #모델의 예측결과 저장할 파일 경로 입력받기
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    #translator에서 사용하는 beam search 크기 설정
    parser.add_argument('-beam_size', type=int, default=5)
    #생성될 sequence의 최대 길이 설정(기본값은 100으로 함)
    parser.add_argument('-max_seq_len', type=int, default=100)
    #'cuda'를 사용할지 여부에 대한 입력
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")


    
    #     parser.parge_args(): 명령줄에서 전달된 인수들을 해석하고 파이썬 객체로 변환
    #opt: 파싱된 명령줄 옵션들의 값을 속성을 가지고 있는 파이썬 객체
    #     -> opt로 명령줄 옵션들에 opt. 으로 쉽게 접근할 수 있음
    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    # pickle 모듈: 파이썬 객체를 파일로 그대로 저장하고 나중에 그대로 불러올 수 있음
    #opt.data_pkl은 바이너리 파일이므로 '(r)eab (b)iary'
    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']

    #TEXT.vocab.stoi를 통해서 현재 vocab의 단어와 맵핑된 고유한 정수를 출력할 수 있음   
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]   #PAD_WORD = '<blank>'
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]

    # target 문장 시작을 나타내는 index
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    # target 문장 끝을 나타내는 index
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]
    #test데이터 준비
    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    #'cuda'가 사용하면 'cuda'를, cpu 사용해야한다면 cpu를 설정하는 device 변수 정의
    device = torch.device('cuda' if opt.cuda else 'cpu')

    #translator 모듈에서 정의한 translator 모델 불러오기
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)  #Translator도 데이터가 존재하는 device에 있어야하기 때문에 .to(device)로 올려줌

    #source 단어에서 unknown인 토큰의 인덱스
    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))

            #vocab.stoi.get(a,b) : a(단어)를 인덱스로 변환, vocab에 없으면 unk_idx로 변환
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]

            """ translator.translate_sentence()가 번역하는 중"""
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))

            #itos : integer to string : 예측된 타겟 단어 인덱스들을 단어로 변환
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            #시작 토큰과 종료토큰 제거
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            #print(pred_line)
            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
