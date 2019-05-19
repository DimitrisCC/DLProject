import json
import glob
from tqdm.auto import tqdm
from texttable import Texttable
import networkx as nx
import igraph as ig
import os
import numpy as np
import shutil
from src.nystrom import Nystrom
import torch
import torch.utils.data as utils


def load_data(ds_name, use_node_labels):
    node2graph = {}
    Gs = []

    with open("../datasets/%s/%s_graph_indicator.txt" % (ds_name, ds_name), "r") as f:
        c = 1
        for line in f:
            node2graph[c] = int(line[:-1])
            if not node2graph[c] == len(Gs):
                Gs.append(nx.Graph())
            Gs[-1].add_node(c)
            c += 1

    with open("../datasets/%s/%s_A.txt" % (ds_name, ds_name), "r") as f:
        for line in f:
            edge = line[:-1].split(",")
            edge[1] = edge[1].replace(" ", "")
            Gs[node2graph[int(edge[0])] - 1].add_edge(int(edge[0]), int(edge[1]))

    if use_node_labels:
        with open("../datasets/%s/%s_node_labels.txt" % (ds_name, ds_name), "r") as f:
            c = 1
            for line in f:
                node_label = int(line[:-1])
                Gs[node2graph[c] - 1].node[c]['label'] = node_label
                c += 1

    labels = []
    with open("../datasets/%s/%s_graph_labels.txt" % (ds_name, ds_name), "r") as f:
        for line in f:
            labels.append(int(line[:-1]))

    labels = np.array(labels, dtype=np.float)
    return Gs, labels


def write_json_files(dataset, Gs, targets):
    node_labels = []
    edge_lists = []
    for G in Gs:
        labels = nx.get_node_attributes(Gs[0], 'label')
        node_labels.append({str(node): value for node, value in labels.items()})
        edge_lists.append([list(edge) for edge in G.edges()])


def generate_synthetic():
    import random
    max_nodes = 200
    min_nodes = 100
    community_num_nodes = 10
    graphs = []
    labels = []
    com_1 = nx.caveman_graph(1, community_num_nodes)
    com_2 = nx.star_graph(community_num_nodes)

    for i in range(500):
        num_nodes = random.randint(min_nodes, max_nodes)
        graph = nx.fast_gnp_random_graph(num_nodes, 0.1)
        graph = nx.disjoint_union(graph, com_1)
        for i in range(num_nodes, graph.number_of_nodes()):
            for j in range(num_nodes):
                if random.random() > 0.9:
                    graph.add_edge(graph.nodes()[i], graph.nodes()[j])
        graphs.append(graph)
        labels.append(1)
        num_nodes = random.randint(min_nodes, max_nodes)
        graph = nx.fast_gnp_random_graph(num_nodes, 0.1)
        for i in range(num_nodes, graph.number_of_nodes()):
            for j in range(num_nodes):
                if random.random() > 0.9:
                    graph.add_edge(graph.nodes[i], graph.nodes[j])
        graphs.append(graph)
        labels.append(0)

    return graphs, labels


def networkx_to_igraph(G):
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    reverse_mapping = dict(zip(range(G.number_of_nodes()), G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    G_ig = ig.Graph(len(G), list(zip(*list(zip(*nx.to_edgelist(G)))[:2])))
    return G_ig, reverse_mapping


def community_detection(G_networkx, community_detection_method):
    G, reverse_mapping = networkx_to_igraph(G_networkx)

    if community_detection_method == "eigenvector":
        c = G.community_leading_eigenvector()
    elif community_detection_method == "infomap":
        c = G.community_infomap()
    elif community_detection_method == "fastgreedy":
        c = G.community_fastgreedy().as_clustering()
    elif community_detection_method == "label_propagation":
        c = G.community_label_propagation()
    elif community_detection_method == "louvain":
        c = G.community_multilevel()
    elif community_detection_method == "spinglass":
        c = G.community_spinglass()
    elif community_detection_method == "walktrap":
        c = G.community_walktrap().as_clustering()
    else:
        c = []

    communities = []
    for i in range(len(c)):
        community = []
        for j in range(len(c[i])):
            community.append(reverse_mapping[G.vs[c[i][j]].index])

        communities.append(community)

    return communities


def compute_communities(graphs, use_node_labels, community_detection_method):
    communities = []
    subgraphs = []
    counter = 1
    coms = []
    for G in graphs:
        c = community_detection(G, community_detection_method)
        coms.append(len(c))
        subgraph = []
        for i in range(len(c)):
            communities.append(G.subgraph(c[i]))
            subgraph.append(counter)
            counter += 1

        subgraphs.append(' '.join(str(s) for s in subgraph))

    return communities, subgraphs


def compute_nystrom(ds_name, use_node_labels, embedding_dim, community_detection_method, kernels):
    if ds_name == "SYNTHETIC":
        graphs, labels = generate_synthetic()
    else:
        graphs, labels = load_data(ds_name, use_node_labels)
    communities, subgraphs = compute_communities(graphs, use_node_labels, community_detection_method)

    print("Number of communities: ", len(communities))
    lens = []
    for community in communities:
        lens.append(community.number_of_nodes())

    print("Average size: %.2f" % np.mean(lens))
    Q = []
    for idx, k in enumerate(kernels):
        model = Nystrom(k, n_components=embedding_dim)
        model.fit(communities)
        Q_t = model.transform(communities)
        Q_t = np.vstack([np.zeros(embedding_dim), Q_t])
        Q.append(Q_t)

    return Q, subgraphs, labels, Q_t.shape


def create_train_val_test_loaders(Q, x_train, x_val, x_test, y_train, y_val, y_test, batch_size):
    num_kernels = Q.shape[2]
    max_document_length = x_train.shape[1]
    dim = Q.shape[1]

    my_x = []
    for i in range(x_train.shape[0]):
        temp = np.zeros((1, num_kernels, max_document_length, dim))
        for j in range(num_kernels):
            for k in range(x_train.shape[1]):
                temp[0, j, k, :] = Q[x_train[i, k], :, j].squeeze()
        my_x.append(temp)

    if torch.cuda.is_available():
        tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.cuda.LongTensor(y_train.tolist())
    else:
        tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.from_numpy(np.asarray(y_train, dtype=np.int64))

    train_dataset = utils.TensorDataset(tensor_x, tensor_y)
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    my_x = []
    for i in range(x_val.shape[0]):
        temp = np.zeros((1, num_kernels, max_document_length, dim))
        for j in range(num_kernels):
            for k in range(x_val.shape[1]):
                temp[0, j, k, :] = Q[x_val[i, k], :, j].squeeze()
        my_x.append(temp)

    if torch.cuda.is_available():
        tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.cuda.LongTensor(y_val.tolist())
    else:
        tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.from_numpy(np.asarray(y_val, dtype=np.int64))

    val_dataset = utils.TensorDataset(tensor_x, tensor_y)
    val_loader = utils.DataLoader(val_dataset, batch_size=1, shuffle=False)

    my_x = []
    for i in range(x_test.shape[0]):
        temp = np.zeros((1, num_kernels, max_document_length, dim))
        for j in range(num_kernels):
            for k in range(x_test.shape[1]):
                temp[0, j, k, :] = Q[x_test[i, k], :, j].squeeze()
        my_x.append(temp)

    if torch.cuda.is_available():
        tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.cuda.LongTensor(y_test.tolist())
    else:
        tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
        tensor_y = torch.from_numpy(np.asarray(y_test, dtype=np.int64))

    test_dataset = utils.TensorDataset(tensor_x, tensor_y)
    test_loader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def save_checkpoint(state, is_best, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_model_file)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def create_numeric_mapping(node_properties):
    """
    Create node feature map.
    :param node_properties: List of features sorted.
    :return : Feature numeric map.
    """
    return {value: i for i, value in enumerate(node_properties)}


def read_node_labels(args):
    """
    Reading the graphs from disk.
    :param args: Arguments object.
    :return identifiers: Hash table of unique node labels in the dataset.
    :return class_number: Number of unique graph classes in the dataset. 
    """
    print("\nCollecting unique node labels.\n")
    labels = set()
    targets = set()
    graphs = glob.glob(args.train_graph_folder + "*.json")
    try:
        graphs = graphs + glob.glob(args.test_graph_folder + "*.json")
    except:
        pass
    for g in tqdm(graphs):
        data = json.load(open(g))
        labels = labels.union(set(list(data["labels"].values())))
        targets = targets.union(set([data["target"]]))
    identifiers = {label: i for i, label in enumerate(list(labels))}
    class_number = len(targets)
    print("\n\nThe number of graph classes is: " + str(class_number) + ".\n")
    return identifiers, class_number


def create_logs(args):
    """
    Creates a dictionary for logging.
    :param args: Arguments object.
    :param log: Hash table for logs.
    """
    log = dict()
    log["losses"] = []
    log["params"] = vars(args)
    return log


def create_features(data, identifiers):
    """
     Creates a tensor of node features. 
    :param data: Hash table with data.
    :param identifiers: Node labels mapping.
    :return graph: NetworkX object.
    :return features: Feature Tensor (PyTorch).
    """
    graph = nx.from_edgelist(data["edges"])
    features = [[1.0 if data["labels"][str(node)] == i else 0.0 for i in range(len(identifiers))] for node in
                graph.nodes()]
    features = np.array(features, dtype=np.float32)
    features = torch.tensor(features)
    return graph, features


def create_batches(graphs, batch_size):
    """
    Creating batches of graph locations.
    :param graphs: List of training graphs.
    :param batch_size: Size of batches.
    :return batches: List of lists with paths to graphs.
    """
    batches = [graphs[i:i + batch_size] for i in range(0, len(graphs), batch_size)]
    return batches


def calculate_reward(target, prediction):
    """
    Calculating a reward for a prediction.
    :param target: True graph label.
    :param prediction: Predicted graph label.
    """
    reward = (target == torch.argmax(prediction))
    reward = 2 * (reward.float() - 0.5)
    return reward


def calculate_predictive_loss(data, predictions):
    """
    Prediction loss calculation.
    :param data: Hash with label.
    :param prediction: Predicted label.
    :return target: Target tensor.
    :prediction loss: Loss on sample.
    """
    target = [data["target"]]
    target = torch.tensor(target)
    prediction_loss = torch.nn.functional.nll_loss(predictions, target)
    return target, prediction_loss
