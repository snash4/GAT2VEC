import os


# TODO: use config file instead?
def get_dataset_name(input_dir):
    _, dataset = os.path.split(input_dir)
    return dataset


def get_adjlist_path(dataset_dir, gtype):
    file_name = get_dataset_name(dataset_dir) + '_' + gtype + '.adjlist'
    return os.path.join(dataset_dir, file_name)


def get_labels_path(dataset_dir):
    return os.path.join(dataset_dir, 'labels_maped.txt')


def get_embedding_path(input_dir, output_dir):
    file_name = get_dataset_name(input_dir) + "_gat2vec.emb"
    return os.path.join(output_dir, file_name)


def get_embedding_path_bip(input_dir, output_dir):
    file_name = get_dataset_name(input_dir) + "_gat2vec_bip.emb"
    return os.path.join(output_dir, file_name)


def get_embedding_path_wl(input_dir, output_dir, tr):
    file_name = get_dataset_name(input_dir) + "_gat2vec_label_" + str(int(tr * 100)) + ".emb"
    return os.path.join(output_dir, file_name)


def get_embedding_path_param(input_dir, output_dir, param, ps, pa):
    file_name = get_dataset_name(input_dir) + "_gat2vec_" + param + "_nwalks_" + str(ps) + str(
        pa) + ".emb"
    return os.path.join(output_dir, file_name)


def get_param_csv_path(dataset, param):
    return dataset + "_paramsens_" + param + "_nwalks_" + ".csv"
