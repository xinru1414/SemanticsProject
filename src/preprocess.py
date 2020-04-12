"""
Jan 2020
Xinru Yan

This program prepares the donation dialogue data ready

Input:
    data_annotate: 297 dialogues annotated with face labels
    data_info: 1,018 dialogue information on donation
    data_full: 1,018 dialogues
Output:
    a csv file with each row containing the 'message_id', 'text' of full conversation and the 'donation_label' (0 for no
    donation, 1 for the persuadee made a donation)
"""
import csv
import pickle
from enum import Enum
import click


class DataInfoHeader(Enum):
    CONVERSATION_ID = 'B2'
    DONATION = 'B6'
    SPEAKER = 'B4'

    @property
    def column_name(self):
        return self.value


class DataFaceHeader(Enum):
    CONVERSATION_ID = 'B2'
    SENT = 'Unit'
    SPEAKER = 'B4'
    FACE_LABEL = 'Poss_labels'
    SEMANTIC_LABEL = 'New Label'

    @property
    def column_name(self):
        return self.value


class Conversation:
    def __init__(self, message_id):
        self.message_id = message_id
        self.sents = []
        self.speakers = []
        self.face_labels = []
        self.semantic_labels = []
        self.donation_label = None

    def add_sent(self, sent):
        #self.sents += [sent]
        self.sents.append([sent])

    def add_speaker(self, speaker):
        self.speakers.append(int(speaker))

    def add_face_label(self, face_label):
        self.face_labels.append(face_label)

    def add_semantic_label(self, semantic_label):
        self.semantic_labels.append(semantic_label)

    def add_donation_label(self, donation_label):
        self.donation_label = donation_label

    def __len__(self):
        assert len(self.sents) == len(self.speakers) == len(self.face_labels) == len(self.semantic_labels), 'length of utterracnes should equal to length of speakers and face labels'
        return len(self.sents)

    @property
    def dict(self):
        d = {'message_id': self.message_id,
             'text': self.sents, #' '.join(self.sents),#self.sents, #' '.join(self.sents),  ,
             'speaker': self.speakers,
             'semantic_label': self.semantic_labels,
             'face_label': self.face_labels,
             'donation_label': self.donation_label}
        return d


def read_data(csv_file, conversations):
    face_labels = set()
    semantic_labels = set()

    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataFaceHeader.CONVERSATION_ID.column_name]
            if message_id not in conversations:
                conversations[message_id] = Conversation(message_id)
            sent = row[DataFaceHeader.SENT.column_name]
            speaker = row[DataFaceHeader.SPEAKER.column_name]

            semantic = row[DataFaceHeader.SEMANTIC_LABEL.column_name]
            semantic = semantic.lower().strip()
            if semantic == '':
                semantic = 'none'
            semantic = semantic.splitlines()[0]
            semantic_labels.add(semantic)

            face = row[DataFaceHeader.FACE_LABEL.column_name]
            face = face.lower().strip()
            if len(face) > 6:
                face = face[:6]
            if face == '':
                face = 'none'
            face_labels.add(face)

            conversations[message_id].add_sent(sent)
            conversations[message_id].add_speaker(speaker)
            conversations[message_id].add_semantic_label(semantic)
            conversations[message_id].add_face_label(face)

    for semantic in semantic_labels:
        print(semantic)

    #l2i = {label: i for i, label in enumerate(list(semantic_labels))}
    #i2l = {i: label for label, i in l2i.items()}

    #print(l2i)

    #conversations = switch_face_label(conversations, l2i)
    return conversations


def switch_face_label(conversations, l2i):
    for key, convo in conversations.items():
        labels = convo.face_labels
        new_labels = [l2i[label] for label in labels]
        convo.face_labels = new_labels
    return conversations


def read_face_label(csv_file):
    conversations = {}
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataFaceHeader.CONVERSATION_ID.column_name]
            if message_id not in conversations:
                conversations[message_id] = Conversation(message_id)
            face_label = row[DataFaceHeader.FACE_LABEL.column_name]
            if len(face_label) > 6:
                face_label = face_label[:5]
            if face_label == '':
                face_label = 'None'
            conversations[message_id].add_sent(face_label)
    return conversations


def write_data(csv_file, conversations):
    with open(csv_file, 'w') as csvfile:
        headers = ['message_id', 'text', 'donation_label']
        writer = csv.DictWriter(csvfile, headers, extrasaction='ignore')
        writer.writeheader()
        for conversation in conversations.values():
            writer.writerow(conversation.dict)


def write_pickle(pickle_file, conversations):
    with open(pickle_file, "wb") as fp:
        pickle.dump(conversations, fp)


def read_pickle(pickle_file):
    with open(pickle_file, "rb") as fp:
        conversations = pickle.load(fp)
    return conversations


def donation(value: str):
    if float(value) > 0:
        label = 1
    else:
        label = 0
    return label


@click.command()
@click.option('-i', '--dataset_info', default='../data/raw/data_info.csv')
@click.option('-f', '--dataset_full', default='../data/raw/data_full.csv')
@click.option('-a', '--dataset_annotate', default='../data/raw/semantic.csv')
@click.option('-o_e', '--output_entire', default='../data/preprocessed/102_prepared_data.pickle')
@click.option('-o_t', '--output_annotate_train', default='../data/preprocessed/102_train_prepared_data.pickle')
@click.option('-o_d', '--output_annotate_dev', default='../data/preprocessed/102_dev_prepared_data.pickle')
@click.option('-o_e', '--output_annotate_test', default='../data/preprocessed/102_test_prepared_data.pickle')
def main(dataset_info, dataset_full, dataset_annotate, output_entire, output_annotate_train, output_annotate_dev, output_annotate_test):

    annotate_conversations = {}
    annotate_conversations = read_data(dataset_annotate, annotate_conversations)

    print(f'annotated dataset {len(annotate_conversations)}')

    with open(f'{dataset_info}', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            message_id = row[DataInfoHeader.CONVERSATION_ID.column_name]
            if message_id in annotate_conversations:
                value = row[DataInfoHeader.DONATION.column_name]
                donation_label = donation(value)
                annotate_conversations[message_id].add_donation_label(donation_label)

    entire = list(annotate_conversations.items())


    assert len(entire) == 102
    train = dict(entire[:82])
    assert len(train) == 82
    dev = dict(entire[82:92])
    assert len(dev) == 10
    test = dict(entire[92:])
    assert len(test) == 10

    print(f'writing to {output_entire}')
    write_pickle(output_entire, entire)
    print(f'writing to {output_annotate_train}')
    write_pickle(output_annotate_train, train)
    print(f'writing to {output_annotate_dev}')
    write_pickle(output_annotate_dev, dev)
    print(f'writing to {output_annotate_test}')
    write_pickle(output_annotate_test, test)


if __name__ == '__main__':
    main()


