import os


# TODO: use config file instead?
def get_dataset_dir(dataset):
    return os.path.join(os.getcwd(), "data", dataset)


def get_adjlist_path(dataset_dir, dataset, gtype):
    file_name = dataset + '_' + gtype + '.adjlist'
    return os.path.join(dataset_dir, file_name)


def get_labels_path(dataset_dir):
    return os.path.join(dataset_dir, 'labels_maped.txt')


def get_embedding_path(dataset):
    file_name = dataset + "_gat2vec.emb"
    return os.path.join("./embeddings", file_name)


def get_embedding_path_bip(dataset):
    file_name = dataset + "_gat2vec_bip.emb"
    return os.path.join("./embeddings", file_name)


def get_embedding_path_wl(dataset, tr):
    file_name = dataset + "_gat2vec_label_" + str(int(tr * 100)) + ".emb"
    return os.path.join("./embeddings", file_name)


def get_embedding_path_param(dataset, param, ps, pa):
    file_name = dataset + "_gat2vec_" + param + "_nwalks_" + str(ps) + str(pa) + ".emb"
    return os.path.join("./embeddings", file_name)


def get_param_csv_path(dataset, param):
    return dataset + "_paramsens_" + param + "_nwalks_" + ".csv"
