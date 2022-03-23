import os
import random
import copy
import json
import argparse
import sys
import numpy as np
import networkx as nx
from tqdm import tqdm

import ndtw

sys.setrecursionlimit(1000000)


def load_nav_graphs(scans):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    if scans is None:
        scans = load_scan()
    graphs = {}
    for scan in scans:
        with open('connectivity/%s_connectivity.json' % scan) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]])
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


class TreeNode(object):
    """The basic node of tree structure"""

    def __init__(self, name, parent=None, step=0):
        super(TreeNode, self).__init__()
        self.name = name
        self.parent = parent
        self.child = {}
        self.step = step

    def __repr__(self) :
        return 'TreeNode(%s)' % self.name


    def __contains__(self, item):
        return item in self.child


    def __len__(self):
        """return number of children node"""
        return len(self.child)

    def __bool__(self, item):
        """always return True for exist node"""
        return True

    def get_child(self, name, defval=None):
        """get a child node of current node"""
        return self.child.get(name, defval)

    def add_child(self, name, obj=None):
        """add a child node to current node"""
        if obj and not isinstance(obj, TreeNode):
            raise ValueError('TreeNode only add another TreeNode obj as child')
        if obj is None:
            obj = TreeNode(name, self.step+1)
        obj.parent = self
        self.child[name] = obj
        return obj

    def del_child(self, name):
        """remove a child node from current node"""
        if name in self.child:
            del self.child[name]

    def find_child(self, path, create=False):
        """find child node by path/name, return None if not found"""
        # convert path to a list if input is a string
        path = path if isinstance(path, list) else path.split()
        cur = self
        for sub in path:
            # search
            obj = cur.get_child(sub)
            if obj is None and create:
                # create new node if need
                obj = cur.add_child(sub)
            # check if search done
            if obj is None:
                break
            cur = obj
        return obj

    def items(self):
        return self.child.items()

    def print_curr_path(self):
        curr = []
        tmp = copy.deepcopy(self)
        while tmp is not None:
            curr.append(tmp.name)
            tmp = tmp.parent
        curr.reverse()
        return curr


def DFS(graph, start, end, source, initial_node, stack, visit, paths, min_len, max_len, inter=30):
    stack.append(start)
    visit.add(start)

    loop_max = 30000
    loop_n = 0
    while (len(stack) > 1 and stack[-1] != source) or (len(stack) == 1):
        loop_n += 1
        if loop_n >= loop_max:
            break

        vertex = stack[-1]
        if vertex == end:
            assert stack[0] == initial_node
            if len(stack) > min_len:
                paths.append(stack.copy())
                if len(paths) >= inter:
                    break
            stack.pop()
            vertex = stack[-1]

        neighbors = graph.neighbors(vertex)
        for w in neighbors:
            if w not in visit:
                visit.add(w)

                try:
                    min_step_len = len(stack) + len(nx.dijkstra_path(graph, w, end))
                except Exception as expt:
                    if "not reachable" in str(expt):
                        pass
                    continue

                if min_step_len <= max_len:
                    DFS(graph, w, end, vertex, initial_node, stack.copy(), visit.copy(), paths, min_len, max_len)
                    # print(paths)
        stack.pop()


def BFS(graph, start, end, initial_node, queue, paths, min_len, max_len, inter=30):
    # tree = TreeNode(start)
    tree = [start]
    queue.append(tree)
    # visit.add(start)

    # near the end
    near_end_list = []
    neighbors = graph.neighbors(end)
    for w in neighbors:
        near_end_list.append(w)

    near_end_list_2 = []
    for wv in near_end_list:
        neighbors = graph.neighbors(wv)
        near = []
        for w in neighbors:
            near.append(w)
        near_end_list_2.append(near)

    loop_max = 5000
    loop_n = 0

    while len(queue) > 0:
        loop_n += 1
        if loop_n >= loop_max:
            break

        if len(paths) >= inter:
            break
        # tree = copy.deepcopy(queue[0])
        tree = queue[0].copy()
        del queue[0]
        # vertex = tree.name
        vertex = tree[-1]
        if vertex == end:
            # current_path = tree.print_curr_path()
            current_path = tree
            assert current_path[0] == initial_node and current_path[-1] == end
            if min_len <= len(current_path) <= max_len:
            # elif (not large and len(current_path) < max_len) or (large and min_len < len(current_path) < max_len):
                paths.append(current_path.copy())
            continue

        # if tree.step >= max_len:
        if len(tree) >= max_len:
            continue

        neighbors = graph.neighbors(vertex)
        for w in neighbors:
            # if w not in visit:
            #     visit.add(w)
            # if tree.find_child(w) is None:
            if w not in tree:
                prior = False
                index_n = 0
                for ind_n, node_l in enumerate(near_end_list_2):
                    if w in node_l:
                        prior = True
                        index_n = ind_n
                        break

                if not prior:
                    # tree = tree.add_child(w)
                    # queue.append(copy.deepcopy(tree))
                    # tree = tree.parent
                    tree2 = tree.copy()
                    tree2.append(w)
                    queue.append(tree2)
                else:
                    # tree = tree.add_child(w)
                    # tree = tree.add_child(near_end_list[index_n])
                    # tree = tree.add_child(end)
                    # queue.append(copy.deepcopy(tree))
                    # tree = tree.parent.parent.parent
                    tree2 = tree.copy()
                    tree2.append(w)
                    tree2.append(near_end_list[index_n])
                    tree2.append(end)
                    queue.append(tree2)


def compute_distances(paths):
    distances = []
    for path in paths:
        dis = 0
        for ind, node in enumerate(path[1:]):
            dis += nx.dijkstra_path_length(G, path[ind], node)
        distances.append(dis)
    return np.array(distances)


def compute_dtw(paths, gt_path, scan, ndtw_criterion):
    ndtws_val = []
    for path in paths:
        ndtw_val = ndtw_criterion[scan](path, gt_path, metric='ndtw')
        ndtws_val.append(ndtw_val)
    return ndtws_val


def repeat_view(origin_path, paths):
    repeat_num = 3
    while len(paths) < 10:
        ind = random.randint(0, len(origin_path) - 1)
        rep = random.randint(1, repeat_num - 1)
        view = origin_path[ind]
        new_path = copy.copy(origin_path)
        for i in range(rep):
            new_path.insert(ind, view)
        paths.append(new_path)


def load_scan():
    scans = []
    with open('connectivity/scans.txt', 'r') as f:
        for line in f.readlines():
            scans.append(line.strip())
    return scans


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--n', dest='n', default=0, type=int, help='n')
    parser.add_argument('--source', type=str, help='input file')
    parser.add_argument('--target', type=str, help='output file')
    args = parser.parse_args()

    scans = []
    with open('connectivity/scans.txt', 'r') as f:
        for line in f.readlines():
            scans.append(line.strip())

    with open(args.source, 'r') as f:
        data = json.load(f)

    graphs = load_nav_graphs(scans)
    min_len = 10
    max_len = 20
    max_num = 30
    error_margin = 3.0

    all_distances = {}
    for scan, G in graphs.items():  # compute all shortest paths
        all_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    print('Init ndtw...')
    ndtw_criterion = ndtw.ndtw_initialize(scans)
    print('Init ndtw finished!')

    aug_data = []
    invalid = 0
    not_use = 0
    for data_i in tqdm(data):
        scan = data_i['scan']
        G = graphs[scan]
        nodes = dict(G.nodes())
        mid_nodes = list(nodes.keys())
        source_node = data_i['path'][0]
        target_node = data_i['path'][-1]
        # aug_data_i = data_i.copy()
        # origin_path = data_i['path']
        all_paths = []

        # near goal
        nears = []
        nears.append(target_node)
        for node in mid_nodes:
            if 0 < all_distances[scan][node][target_node] <= error_margin:
                nears.append(node)

        if len(nears) == 0:
            aug_data.append([])
            invalid += 1
            print(invalid)
            continue
        # inter = max_num // len(nears)
        min_len = len(data_i['path'])

        for n_ind, node in enumerate(nears):
            queue = []
            paths = []
            stack = []
            visit = set()
            # DFS(G, source_node, node, source_node, source_node, stack, visit, paths, min_len, max_len, inter=inter)
            if n_ind == 0:
                BFS(G, source_node, node, source_node, queue, paths, 16, max_len, inter=max_num - len(all_paths))
            else:
                BFS(G, source_node, node, source_node, queue, paths, min_len, max_len, inter=max_num-len(all_paths))
            all_paths += paths.copy()
            if len(all_paths) >= max_num:
                break

        if len(all_paths) == 0:
            aug_data.append([])
            invalid += 1
            print(invalid)
            continue
        distances = compute_distances(all_paths)
        ndtws_val = compute_dtw(all_paths, data_i['path'], scan, ndtw_criterion)
        aug_paths = []
        for path_ind, path in enumerate(all_paths):
            aug_path = {}
            aug_path['path'] = path
            aug_path['distance'] = distances[path_ind]
            aug_path['ndtw'] = ndtws_val[path_ind]
            aug_paths.append(aug_path)
        aug_data.append(aug_paths)

    print('origin data len: {}'.format(len(data)))
    print('new data len: {}'.format(len(aug_data)))

    with open(args.target, 'w') as f:
        json.dump(aug_data, f, indent=4)

