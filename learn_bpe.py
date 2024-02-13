#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
Unlike the original BPE, it does not compress the plain text, but can be used to reduce the vocabulary
of a text to a configurable number of symbols, with only a small increase in the number of tokens.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

"""
BPE(Byte-Pair Encoding)
출현 빈도수가 가장 높은 Byte쌍(연속적인 두개의 알파벳?)을 1 byte로 replace하는 data compression 방법
출현 빈도수가 가장 높은 token들을 merge해가면서 최종 token을 만들어내는 방식
"""

################
# 순서는
# update_vocab -> get_pair_statistics -> prune_stats -> replace_pair -> update_pair_statistics -> prune_stats?
################

from __future__ import unicode_literals

import os
import sys
import inspect
import codecs
import re
import copy
import warnings
from collections import defaultdict, Counter


def update_vocabulary(vocab, file_name, is_dict=False):
    """Read text and return dictionary that encodes vocabulary
    """

    #vocab = Counter()
    with codecs.open(file_name, encoding='utf-8') as fobj:
        for i, line in enumerate(fobj):
            #is_dict가 뭐임???? line이 dictionary로 어떻게 있다는 거지?
            if is_dict:
                try:
                    #'\r': 행의 가장 앞으로 이동시킴 / '\r\n': 커서를 앞으로 보내고 엔터치기
                    word, count = line.strip('\r\n ').split(' ')
                except:
                    print('Failed reading vocabulary file at line {0}: {1}'.format(i, line))
                    sys.exit(1)
                vocab[word] += int(count)
            else:
                for word in line.strip('\r\n ').split(' '):
                    #단어가 있으면
                    if word: #근데 왜 if를..? 어차피 for문 끝나면 단어 없는 거 아닌가?
                        vocab[word] += 1 #vocab
    return vocab


def update_pair_statistics(pair, changed, stats, indices):
    #update_pair_statistics(most_frequent, changes, stats, indices) 이렇게 쓰임

    #most_frequent:tuple || ('a','b') byte pair
    #changed:list || [(j, new_word, word, freq),,,] :merge돼서 업데이트된 단어
    #stats: dict || {('a', 'l'): freq, ('l', 'i'):freq ,,, }
    #indices: dict || {('a','l') : defaultdict({0:1, 1:1, 2:1})} byte pair 쌍을 가진 단어의 인덱스와 그 단어의 빈도수
    """Minimally update the indices and frequency of symbol pairs

    if we merge a pair of symbols, only pairs that overlap with occurrences
    of this pair are affected, and need to be updated.
    """

    # stats = {('a', 'l'): freq, ('l', 'i'):freq ,,, }
    # indices = {('a','l') : defaultdict({0:1, 1:1, 2:1})} 
    #입력된 byte pair 정보 초기화
    stats[pair] = 0
    indices[pair] = defaultdict(int) 
    first, second = pair #first = 'a' second = 'l'
    new_pair = first+second #new_pair = 'al'
    for j, word, old_word, freq in changed:   #old_word = ('a', 'l', 'i', 'c', 'e')
                                              #word = ('al', 'i', 'c', 'e')
        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i) #i번째 요소부터 검색
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word)-1 and old_word[i+1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i-1:i+1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word)-2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i+2] != first or i >= len(old_word)-3 or old_word[i+3] != second:
                        nex = old_word[i+1:i+3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i-1:i+1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word)-1 and word[i+1] != new_pair:
                nex = word[i:i+2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1


def get_pair_statistics(vocab): #vocab: #{('a', 'l', 'i', 'c', 'e</w>') : count, ,,,} 이러한 {단어의 character들 나열: count} dict 형태
    """Count frequency of all symbol pairs, and create index"""
    """
    모든 symbol pair들의 frequency 계산
    """

    # data structure of pair frequencies
    #defaultdict(int)를 하면 값을 지정하지 않은 key는 0으로 지정됨
    stats = defaultdict(int) 

    #index from pairs to words
    #lambda로 기본값 defaultdict() 지정: indices[k]만하고 v입력 안하면 defaultdict( { k: defaultdict({ }) } ) 가 됨
    indices = defaultdict(lambda: defaultdict(int))

    #앞에서부터 두 글자씩 짝 지어서 byte pair 만들기
    for i, (word, freq) in enumerate(vocab): #word: ('a', 'l', 'i', 'c', 'e</w>') // freq: 단어 alice의 frequency
        prev_char = word[0] #첫번째 단어
        for char in word[1:]:
            stats[prev_char, char] += freq #stats = {('a', 'l'): freq, ('l', 'i'):freq ,,, }
            indices[prev_char, char][i] += 1 #indices = {('a','l') : defaultdict({0:1, 1:1, 2:1})} // 0번째 단어에는 ('a','l')쌍 1개 / 1번째 단어에도 1개,,, /// ('a','l')쌍이 vocab 몇번째 단어에 존재하는지
            prev_char = char

    return stats, indices




def replace_pair(pair, vocab, indices):
    #모든 byte쌍들 중에서 "출현 빈도가 가장 높은" byte쌍을 하나로 합치기 ex) ('A', 'B') = ('AB')
    #    -> 두 쌍의 출현빈도가 가장 높다면, 같이 묶여있는 단어일 확률이 높으므로 // BPE는 원래 data compression 방법 중 하나였다고 함

    #   replace_pair(most_frequent, sorted_vocab, indices) 이렇게 쓰임
    #   sorted_vocab는 빈도수(count)가 내림차순 정렬된 sorted로 반환된 list
    #   sorted_vocab = [(('a', 'l', 'i', 'c', 'e</w>') : count) , ,,,} 이런 식

    """Replace all occurrences of a symbol pair ('A', 'B') with a new symbol 'AB'"""
    #ex) pair = ('a', 'l') 
    first, second = pair  #first = 'a', second = 'l'
    pair_str = ''.join(pair) #'al'
    """
    cmd에서 처럼...?
    """
    pair_str = pair_str.replace('\\','\\\\') #'\' -> '\\'로 변환함 #'\'가 어디에 쓰이지?
    changes = []
    #?<!: 부정형 후방탐색
    #\S: 공백이 아닌 것
    #?!: 부정형 전방탐색
    #문자열 시작이나???? 앞쪽에 공백이 있는 경우 // 'first second'는 정규표현식 적용 x  // 뒤쪽에 공백이 있어야한다.
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    #버전 번호의 숫자튜플?
    if sys.version_info < (3, 0):
        iterator = indices[pair].iteritems()
    else:
        iterator = indices[pair].items() #byte pair쌍이 있는 모든 단어 {인덱스:freq}가 담긴 딕셔너리
    for j, freq in iterator:
        if freq < 1:
            continue
        #input으로 들어오는 vocab는 딕셔너리 아니고 sorted로 내림차순 정렬된 list라서 인덱스 접근 가능
        #   sorted_vocab = [(('a', 'l', 'i', 'c', 'e</w>'), count) , ,,,} 이런 식
        word, freq = vocab[j] #word = ('a', 'l', 'i', 'c', 'e</w>') freq = count
        new_word = ' '.join(word) #a l i c e
        new_word = pattern.sub(pair_str, new_word) #new_word에서 pattern에 일치하는 모든 부분을 pair_str('al')로 대체
        new_word = tuple(new_word.split(' ')) #('al', 'i', 'c', 'e')

        vocab[j] = (new_word, freq) #vocab 업데이트
        changes.append((j, new_word, word, freq)) #j: 단어인덱스

    return changes

def prune_stats(stats, big_stats, threshold):
    #밑이 무슨 말인지 모르겠음 해석안됨
    """Prune statistics dict for efficiency of max()

    The frequency of a symbol pair never increases, so pruning is generally safe        
    (until we the most frequent pair is less frequent than a pair we previously pruned)
    big_stats keeps full statistics for when we need to access pruned items
    """
    #vocab에서 자주 등장하지 않는 희소한 단어를 pruning함 -> threshold frequency를 설정해서 threshold보다 작으면 그 pair 제거
    #stats 형상: stats = {('l', 'i'):freq ,,, }
    #          big_stats = {('a', 'l'): freq, ('l', 'i'):freq ,,, }
    for item,freq in list(stats.items()):
        if freq < threshold:
            del stats[item]
            if freq < 0: #freq가 0보다 작을 수가 있음?
                big_stats[item] += freq
            else: #freq >= 0일 때
                big_stats[item] = freq  #big_stats: 나중에 pruned item에 접근해야할 일이 생길 때를 대비해 저장하는 dictionary인데 big_stats는 stats의 deepcopy아닌가?
                                        #똑같은 값을 다시 쓰는 거 아님?


def learn_bpe(infile_names, outfile_name, num_symbols, min_frequency=2, verbose=False, is_dict=False, total_symbols=False):
    """Learn num_symbols BPE operations from vocabulary, and write to outfile.
    """
    sys.stderr = codecs.getwriter('UTF-8')(sys.stderr.buffer)
    sys.stdout = codecs.getwriter('UTF-8')(sys.stdout.buffer)
    sys.stdin = codecs.getreader('UTF-8')(sys.stdin.buffer)

    #vocab = get_vocabulary(infile, is_dict)
    vocab = Counter()
    for f in infile_names:
        sys.stderr.write(f'Collecting vocab from {f}\n')
        #반환되는 vocab는 단어별 빈도수를 담은 Counter : Counter({'alice':3, 'jason':5,,,})
        vocab = update_vocabulary(vocab, f, is_dict) 

    """1. 단어를 character 단위로 분리시키고 단어의 끝을 나타내는 </w>를 단어의 마지막 character에 추가
          vocab = Counter({'alice':3, 'jason':5,,,})
                            |
                            V
          vocab = {('a', 'l', 'i', 'c', 'e</w>') : count, ,,,} 이런 형태가 됨
    """
    #dict([ (a,b), (c,d) ]) -> {a:b, c:d}가 됨!
    vocab = dict([(tuple(x[:-1]) + (x[-1]+'</w>',) ,y) for (x,y) in vocab.items()]) #x: 단어 y:frequency
    

    """2. 단어별 frequency를 기준으로 내림차순 정렬
          sorted_vocab: 출현빈도가 가장 큰 단어부터 내림차순 정렬된 list
          sorted_vocab = [(('a', 'l', 'i', 'c', 'e</w>'), count),,,,,] """
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    
    """3. 단어별로 2개씩 짝을 지어서 byte pair 만들기
          stats = {('a', 'l'): freq, ('l', 'i'): freq, ('i','c'): freq ,,,}
          indices = {('a','l') : defaultdict({0:1, 1:1, 2:1})}   -> ('a','l') byte pair가 존재하는 단어들의 인덱스를 담고있음
    """
    stats, indices = get_pair_statistics(sorted_vocab)
    
    """4. rare하게 등장하는 단어들을 prune하는데 prune한 단어들을 담는 big_stats -> 잘못 prune했을 경우 되돌아오기 위함.
    """
    big_stats = copy.deepcopy(stats) #deepcopy하면 완전히 새로운 객체가 생성, 완전한 복사

    if total_symbols: #total_symbols ??? 계산빨리해주는 애?
        uniq_char_internal = set()
        uniq_char_final = set()

        for word in vocab:    #vocab = {('a', 'l', 'i', 'c', 'e</w>') : count, ,,,} 이런 형태
            for char in word[:-1]:
                uniq_char_internal.add(char) #uniq_char_internal = {'a','l','i','c'} #set
            uniq_char_final.add(word[-1]) #uniq_char_final = {'e</w>'} #set
        sys.stderr.write('Number of word-internal characters: {0}\n'.format(len(uniq_char_internal)))
        sys.stderr.write('Number of word-final characters: {0}\n'.format(len(uniq_char_final)))
        #byte pair merge 작업을 통해 vocab size가 줄어들고 있다는 것을 알리기 위한 표시?
        sys.stderr.write('Reducing number of merge operations by {0}\n'.format(len(uniq_char_internal) + len(uniq_char_final)))
        #num_symbols는 새롭게 만들어진 symbol수?
        # a, l, i, c, e = al, i, c , e 
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)


    sys.stderr.write(f'Write vocab file to {outfile_name}')
    with codecs.open(outfile_name, 'w', encoding='utf-8') as outfile:
        # version 0.2 changes the handling of the end-of-word token ('</w>');
        # version numbering allows bckward compatibility

        outfile.write('#version: 0.2\n')
        # threshold is inspired by Zipfian assumption, but should only affect speed
        """
        여기서부터 진짜 byte pair encoding
        1. threshold를 설정해서 rare한 단어는 무시
        """
        #byte pair의 가장 큰 빈도수값 / 10이 threshold
        threshold = max(stats.values()) / 10
        
        """새로 생성된 symbol수만큼 반복?
        """
        for i in range(num_symbols):
            if stats:
                #가장 최빈 pair을 찾을 때 1st 기준) freq, 2nd 기준) byte pair
                #반환값은 가장 최고로 빈도수가 큰 byte pair (딕셔너리의 key값이 반환됨)
                """most_frequent = ('a','l') : 출현빈도가 가장 큰 byte pair
                """
                most_frequent = max(stats, key=lambda x: (stats[x], x)) 

            # we probably missed the best pair because of pruning; go back to full statistics
            #최고로 빈도수가 큰 byte pair가 threshold를 못 넘을 때?
            if not stats or (i and stats[most_frequent] < threshold):
                prune_stats(stats, big_stats, threshold)
                stats = copy.deepcopy(big_stats)
                most_frequent = max(stats, key=lambda x: (stats[x], x))
                # threshold is inspired by Zipfian assumption, but should only affect speed
                threshold = stats[most_frequent] * i/(i+10000.0)
                prune_stats(stats, big_stats, threshold)

            """stats 형상은 stats = {('a', 'l'): freq, ('l', 'i'): freq, ('i','c'): freq ,,,}"""
            #다 비슷비슷하게 등장하는 애들일 때 stop?
            if stats[most_frequent] < min_frequency:
                sys.stderr.write(f'no pair has frequency >= {min_frequency}. Stopping\n')
                break

            if verbose:
                sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(
                    i, most_frequent[0], most_frequent[1], stats[most_frequent]))
            outfile.write('{0} {1}\n'.format(*most_frequent))

            #vocab update
            changes = replace_pair(most_frequent, sorted_vocab, indices)
            update_pair_statistics(most_frequent, changes, stats, indices)
            stats[most_frequent] = 0
            if not i % 100:
                prune_stats(stats, big_stats, threshold)

