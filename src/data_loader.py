'''
March 2020
Xinru Yan
'''
import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from preprocess import Conversation
import config


class Examples:
    def __init__(self, utterances, semantic_labels, speakers, face_labels):
        self.utterances = utterances
        self.semantic_labels = semantic_labels
        self.speakers = speakers
        self.face_labels = face_labels

        assert len(self.utterances) == len(self.semantic_labels) == len(self.speakers) == len(self.face_labels), 'There must be the same number of utterances, semantic labels and face labels'

    def __add__(self, other: 'Examples'):
        assert isinstance(other, Examples), f'You can only add two Example objects together, not {type(other)} and Example'
        return Examples(self.utterances + other.utterances, self.semantic_labels + other.semantic_labels, self.speakers + other.speakers, self.face_labels + other.face_labels)

    def __len__(self):
        return len(self.utterances)

    def __iter__(self):
        return iter([self.utterances, self.semantic_labels, self.speakers, self.face_labels])

    def shuffled(self):
        c = list(zip(self.utterances, self.semantic_labels, self.speakers, self.face_labels))
        np.random.shuffle(c)
        return Examples(*zip(*c))


class Dataloader:
    def __init__(self, config):
        self.config = config

        self.UNK = '<UNK>'
        self.PAD = '<PAD>'

        self.train_file_path = self.config.train_file_path
        self.dev_file_path = config.dev_file_path
        self.test_file_path = self.config.test_file_path

        self.w2i, self.i2w, self.sl2i, self.si2l, self.fl2i, self.fi2l = self.build_vocabs(self.train_file_path, self.dev_file_path, self.test_file_path)

        self.UNK_IDX = self.w2i[self.UNK]
        self.PAD_IDX = self.w2i[self.PAD]

        self.vocab_size = len(self.w2i)
        self.face_size = len(self.fi2l)
        self.semantic_size = len(self.sl2i)
        self.speaker_size = 2
        print(f'vocab size is {self.vocab_size}, semantic label size is {self.semantic_size}, speaker size is {self.speaker_size}, face label size is {self.face_size}')

        print(self.fl2i, self.fi2l)
        print(self.sl2i, self.si2l)

        print(f'padding')
        self.train_examples = self.pad_sentences(self.load_data(self.train_file_path))
        self.dev_examples = self.pad_sentences(self.load_data(self.dev_file_path))
        self.test_examples = self.pad_sentences(self.load_data(self.test_file_path))

    @staticmethod
    def read_pickle(file):
        with open(file, "rb") as fp:
            conversations = pickle.load(fp)
        return conversations

    def build_vocabs(self, *file_paths):
        words_set = set()
        face_set = set()
        semantic_set = set()

        for file in file_paths:
            convos = self.read_pickle(file)
            for conv_id, value in convos.items():
                semantic_labels = value.semantic_labels
                face_labels = value.face_labels
                sents = value.sents
                for face_label in face_labels:
                    face_set.add(face_label)
                for semantic_label in semantic_labels:
                    semantic_set.add(semantic_label)
                for sent in sents:
                    sent = ' '.join(sent)
                    sent = word_tokenize(sent)
                    sent = ' '.join(sent)
                    for word in sent.strip().split():
                        words_set.add(word)

        word_list = list(words_set)
        w2i = {word: i for i, word in enumerate(word_list)}
        w2i[self.UNK] = len(w2i)
        w2i[self.PAD] = len(w2i)
        i2w = {i: word for word, i in w2i.items()}

        fl2i = {label: i for i, label in enumerate(list(face_set))}
        fi2l = {i: label for label, i in fl2i.items()}

        sl2i = {label: i for i, label in enumerate(list(semantic_set))}
        si2l = {i: label for label, i in sl2i.items()}

        return w2i, i2w, sl2i, si2l, fl2i, fi2l

    def load_data(self, *filepaths) -> Examples:
        total_utterances = []
        total_semantic_labels = []
        total_face_labels = []
        total_speakers = []

        for file in filepaths:
            convos = self.read_pickle(file)
            for conv_id, value in convos.items():
                semantics = []
                semantic_labels = value.semantic_labels
                for semantic_label in semantic_labels:
                    semantics.append(self.sl2i[semantic_label])
                faces = []
                face_labels = value.face_labels
                for face_label in face_labels:
                    faces.append(self.fl2i[face_label])
                speakers = value.speakers
                sents = value.sents
                utterances = []
                for sent in sents:
                    utterance = []
                    sent = ' '.join(sent)
                    sent = word_tokenize(sent)
                    sent = ' '.join(sent)
                    for word in sent.strip().split():
                        utterance.append(self.w2i[word] if word in self.w2i else self.UNK_IDX)
                    utterances.append(utterance)

                assert len(faces) == len(semantics)

                total_semantic_labels.extend(semantics)
                total_face_labels.extend(faces)
                total_speakers.extend(speakers)
                total_utterances.extend(utterances)
        return Examples(utterances=total_utterances, semantic_labels=total_semantic_labels, speakers=total_speakers, face_labels=total_face_labels)

    def pad_sentences(self, examples: Examples) -> Examples:
        semantic_labels = examples.semantic_labels
        face_labels = examples.face_labels
        speakers = examples.speakers
        sents = examples.utterances

        max_len = max(len(sent) for sent in sents)

        padded_sents = [sent if len(sent) == max_len else sent + [self.PAD_IDX] * (max_len - len(sent)) for sent in sents]
        assert [len(sent) == max_len for sent in padded_sents]
        return Examples(utterances=padded_sents, semantic_labels=semantic_labels, speakers=speakers, face_labels=face_labels)


def batch(examples: Examples, batch_size: int):
    batches = []
    batch_num = int(np.ceil(len(examples) / float(batch_size)))
    for i in range(batch_num):
        batch_sents = examples.utterances[i*batch_size:(i+1)*batch_size]
        batch_semantic_labels = examples.semantic_labels[i * batch_size:(i + 1) * batch_size]
        batch_speakers = examples.speakers[i * batch_size:(i + 1) * batch_size]
        batch_face_labels = examples.face_labels[i*batch_size:(i+1)*batch_size]
        batch_data = Examples(batch_sents, batch_semantic_labels, batch_speakers, batch_face_labels)
        batches.append(batch_data)

    for b in batches:
        if len(b) == batch_size:
            yield b


def main():
    dl = Dataloader(config)
    b = batch(dl.train_examples, config.batch_size)

    for bb in b:
        print(len(bb.utterances))
        print(len(bb.semantic_labels))
        print(len(bb.speakers))
        print(len(bb.face_labels))
        break


if __name__ == '__main__':
    main()
