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
bpe_init_ -> process_line -> recursive_split -> check_vocab_and_split -> encode -> isolate_glossary -> segment_tokens -> segment
"""


class BPE(object):
                                        # '@@'는 subunit임 구분자? close@@d ?? 
    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):

        #codes: 파일 객체

        #codes가 가리키는 파일에서 파일 포인터를 파일 시작부분으로 이동시킴
        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        
        #enumerate(codes) : 각 줄 인덱스와 함께 파일 객체의 한 줄 한 줄씩 읽어올 수 있음
            #'\r\n': 커서를 앞으로 보내고 엔터치기
        #[(a,b), (c,d),,,] : 한 문장 속 단어들의 공백 기준 split
            
            #merges????
        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        #한 문장에서 각 단어별로
        for i, item in enumerate(self.bpe_codes):
            #반드시 두 개의 subword unit으로 이루어져 있어야 함?
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        #{(a,b): 5, (c,d):4 ,,,} ??
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        #{ab: (a,b)}
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else [] #분리되지 않을 용어? 모르겠음

        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

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

        return out

    def segment(self, sentence, dropout=0):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
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
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
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
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    word = tuple(word)
    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            #'</w>' 제외해줘야하니까
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return
    
    # vocab에는 seperator가 추가된 애들이 있는건가?
    # lowest = low + est // vocab에 있는 subunit들이 나올 때까지 split 반복
    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

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
    #vocabulary 읽는 함수 -> 어디에서도 안쓰임
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
