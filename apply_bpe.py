#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Rico Sennrich

"""Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import re
import warnings
import random

"""
BPE(Byte-Pair Encoding)
출현 빈도수가 가장 높은 Byte쌍(연속적인 두개의 알파벳?)을 1 byte로 replace하는 data compression 방법
출현 빈도수가 가장 높은 token들을 merge해가면서 최종 token을 만들어내는 방식
"""

"""
순서는
bpe_init_ -> process_line -> recursive_split -> check_vocab_and_split -> encode -> isolate_glossary -> segment_tokens -> segment
"""

class BPE(object):
                                        
    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):
        #codes: 파일 객체
        #merges: 병합횟수 : -1로 설정하면 모든 가능한 subword unit을 생성하여 단어를 분리시킨다고 함
        #seperator: subword끼리 merge했을 때 구분할 수 있는 구분자 '@@'사용: subword a와 b가 결합했을 때 a@@b로
        #gloassaries: BPE를 통해 subword unit으로 분리시킬 때 분리되지 않아야할 용어들

        #codes가 가리키는 파일에서 파일 포인터를 파일 시작부분으로 이동시킴
        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        #firstline이 #version으로 시작하면 버전정보 출력하고
        if firstline.startswith('#version:'):
            #version 정보를 파싱하여 tuple형태로 저장
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            #version 정보가 없으면 버전 추가
            self.version = (0, 1)
            #codes가 가리키는 파일에서 파일 포인터를 파일 시작부분으로 이동시킴
            codes.seek(0)

        
        #enumerate(codes) : 파일객체도 enumerate 사용해서 각 줄 인덱스와 함께 파일 객체의 한 줄 한 줄씩 읽어올 수 있음
            #'\r\n': 커서를 앞으로 보내고 엔터치기
        #[(a,b), (c,d),,,] : 한 문장 속 단어들의 공백 기준 split
        #파일에서 한 줄씩 읽어와서 처리n < merges or merges == -1 일 때 적용: 모든 코드를 파싱하거나 현재 읽은 줄 인덱스가 merges보다 작을 때
        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        #각 bpe 코드가 두 개의 subword unit으로 이루어져있는지 검사
        for i, item in enumerate(self.bpe_codes):
            #반드시 두 개의 subword unit으로 이루어져 있는 파일객체이므로 아니라면 error 발생시킴
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        # 중복된 bpe 코드를 여러번 연산하는 것은 비효율적이므로
        # reversed 하는 이유?
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        #selg.bpe_codes의(pair[0] + pair[1], pair)을 key로 사용하는 딕셔너리 
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        #glossaries 처리, 주어지지 않았으면 빈 리스트 반환
        self.glossaries = glossaries if glossaries else [] 
        #glossaries 각 항목을 ^문자$ 패턴으로 리스트 각 항목을 |로 연결하여 regex 변환
        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None
        #중복 계산을 피하기 위한 cache, 밑에서 if문으로 이미 연산한 건 넘어감
        self.cache = {}

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        #맨 왼쪽 '\r\n '
        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        #앞부분에 공백있었으면 out에 추가
        if leading_whitespace:
            out += line[:leading_whitespace]


        out += self.segment(line, dropout)
        #뒷부분 공백 제거
        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        #뒷부분 공백 있었으면 out에 추가
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]
        #공백이 모두 유지된 output이 반환된
        return out

    def segment(self, sentence, dropout=0):
        #입력문장을 공백 기준으로 split 시켜서 토큰들을 segment화
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        #공백으로 join 시켜서 이어붙임
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        #BPE Encoding을 사용해서 segment화
        """segment a sequence of tokens with BPE encoding"""

        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        #여기서 BPE class의 encode를 사용해서 subword unit으로 분리
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]
            #새로운 단어 만들어졌으면 self.seperator @@를 추가해서 구분해줌
            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output
    #주어진 단어에서 glossaries는 분리해야함: 밑 isolate_glossart 이용
    def _isolate_glossaries(self, word):
        word_segments = [word]
        #self.gloassaries(분리되지 않을 용어를 담은 리스트)에서 순회하면서 isolate_glossary() 적용
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """
    #캐시에 단어 있으면 캐시 반환
    if not dropout and orig in cache:
        return cache[orig]
    
    #glossaries에 정의된 용어랑 match되면 원래 단어 반환
    if glossaries_regex and glossaries_regex.match(orig):
        cache[orig] = (orig,)
        return (orig,)
    
    #단어 한 글자면 그대로 반환
    if len(orig) == 1:
        return orig
    
    #version에 따라서 BPE Encoding 진행
    if version == (0, 1):
        word = list(orig) + ['</w>']  #character 단위로 다 segment 하고 끝에 <\w>추가
    elif version == (0, 2): # more consistent handling of word-final segments
        #마지막 단어 분리하고 </w> 추가
        word = list(orig[:-1]) + [orig[-1] + '</w>']
    else:
        raise NotImplementedError

    #단어 1개만 남을 때까지
    while len(word) > 1:
        
        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        #merge하고자 하는 bigram과 같은 pair 모두 찾기
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        #합치기
        bigram = ''.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            #merge되기 전 이전 단어들 ('i','c','e')
            new_word.extend(word[i:j]) # all symbols before merged pair
            #merge된 애들 ('al')
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

    # don't print end-of-word symbols
    #단어끝을 나타내는 나타내는 </w>이 있으면 마지막 문자 제거
    if word[-1] == '</w>':
        word = word[:-1]
    #단어 끝에 </w>가 중복으로 있을 경우를 고려한 것
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    #단어를 tuple 형태로 변환
    word = tuple(word)
    if vocab:
        #단어가 vocab에 있는지 확인, 있으면 그 단어 bpe 코드로 segmet
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""
    #final은 각 segmet 끝에 </w>가 붙어있는지 여부를 나타내는 변수인 것 같음

    try:
        #final이 True인 경우에는 </w>가 뒤에 붙은 bpe 코드 찾기
        if final:
            left, right = bpe_codes[segment + '</w>']
            #'</w>' 제외해줘야하니까 right에서 제거
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return
    
    # lowest = low + est // vocab에 있는 subunit들이 나올 때까지 split 반복
    # recursive하게 split한 left 부분이 vocab에 있는지 확인, 있으면 그대로 반환하면 됨
    if left + separator in vocab:
        yield left
    #없다면 더 작은 단위로 recursive하게 다시 분할 진행
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    #final이 True인 경우, right이 사전에 있는지 확인
    #final이 False라면, right에 구분자 @@추가해서 vocab에 있는지 확인 -> 없다면 다시 recursive하게 split 진행해야함
    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    #각 분할된 segment가 vocab에 있는지 확인하고, vocab에 속하지 않는 segment들은 다시 더 작은 segment로 분할
    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            #속하지 않으면 더 작게 분할
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    #마지막 글자는 따로 처리해줘야함
    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out


def read_vocabulary(vocab_file, threshold):
    #vocabulary 읽는 함수 -> 어디에서도 안쓰이는데 왜 있는지 의문
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        #threshold가 없거나 threshold보다 빈도수가 큰 단어들만 vocab에 저장
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.

    Returns a list of subwords. In which all 'glossary' glossaries are isolated 

    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    # regex equivalent of (if word == glossary or glossary not in word)

    #분할하면 안되는 용어 glossary를 만나면 분리
    #glossary없으면 그대로 반환
    if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
        return [word]
    #있을 경우 처리
    else:
        #glossary 기준으로 split
        segments = re.split(r'({})'.format(glossary), word)
        segments, ending = segments[:-1], segments[-1]
        #빈 문자열 filter
        segments = list(filter(None, segments)) # Remove empty strings in regex group.
        #뒤에 오는 부분은 따로 추가
        return segments + [ending.strip('\r\n ')] if ending != '' else segments
