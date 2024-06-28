import copy
import bert_embeddings
from datasets import load_dataset
delidata_corpus = load_dataset("gkaradzhov/DeliData")


def create_windows(min_window_size=4,max_window_size=20):
    groups = list(delidata_corpus.keys())
    all_windows = []
    window = []
    for m in delidata_corpus[groups[0]]:
        # print(m['message_type'],m['origin'], m['original_text'],m['annotation_type'])
        if(m['message_type'] == 'INITIAL'): window = []
        window.append(m)
        if(len(window)>max_window_size):
            window.pop(0)
        if(len(window)>=min_window_size):
            all_windows.append(copy.deepcopy(window))
    return all_windows

def last_label(window):
    return window[-1]['annotation_type'], window[-1]['annotation_target']

def bert_to_csv():
    for m in delidata_corpus['train']:
        print(m['origin'], m['original_text'],m['annotation_type'])

# bert_to_csv()
test = ["hi", "this is a test"]
print(bert_embeddings.get_BERT(test))
# all_windows = create_windows()
# print(len(all_windows))
# for i in range(100):
#     print(i, last_label(all_windows[i]))