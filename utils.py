import numpy as np
from random import choice
from random import shuffle
# from pykeops.torch import generic_argkmin
import torch.nn as nn
import torch
import math
from collections import deque
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def move_i_first(index, i):
    if (index == i).nonzero()[..., 0].shape[0]:
        inx = (index == i).nonzero()[0][0]
    else:
        index[-1] = i
        inx = (index == i).nonzero()[0][0]
    if inx > 1:
        index[1:inx+1], index[0] = index[0:inx].clone(), index[inx].clone() 
    else:
        index[inx], index[0] = index[0].clone(), index[inx].clone() 
    return index

def sparse_mx_to_torch_sparse_tensor(sparse_mx, is_L=False):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sp.csr_matrix(sparse_mx)
    # Rescale Laplacian and store as a TF sparse tensor. Copy to not modify the shared L.

    # if is_L:  # block by lsh
    #     sparse_mx = rescale_L(sparse_mx, lmax=2)

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

## get neighbors index of each center node, with fixed number 8
def get_adj(A):
    
    kernal_size = 9
    A_temp = []
    for x in A:
        x.data = np.ones(x.data.shape)
        # build symmetric adjacency matrix
        x = x + x.T.multiply(x.T > x) - x.multiply(x.T > x)
        #x = x + sp.eye(x.shape[0])
        A_temp.append(x.astype('float32'))
    A_temp = [normalize(x) for x in A_temp]
    A = [sparse_mx_to_torch_sparse_tensor(x) for x in A_temp]

    Adj = []
    for adj in A:
        index_list = []
        for i in range(adj.shape[0]): #
            index = (adj._indices()[0] == i).nonzero().squeeze()
            if index.dim() == 0:
                index = index.unsqueeze(0)
            index1 = torch.index_select(adj._indices()[1], 0, index[:kernal_size-1])
            #index1 = move_i_first(index1, i)
            index_list.append(index1)
        index_list.append(torch.zeros(kernal_size-1, dtype=torch.int64)-1)
        index_list = torch.stack([torch.cat([i, i.new_zeros(
            kernal_size - 1 - i.size(0))-1], 0) for inx, i in enumerate(index_list)], 0)
        Adj.append(index_list)
    return Adj

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

# get verts connection matrix n*n , elements is of {0,1}
def get_vert_connectivity(mesh_v, mesh_f):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12."""

    vpv = sp.csc_matrix((len(mesh_v),len(mesh_v)))

    # for each column in the faces...
    for i in range(3):
        IS = mesh_f[:,i]
        JS = mesh_f[:,(i+1)%3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv

###########################################################

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt(np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    return target

def get_target(vertice, face, size):
    target = init_regul(vertice, face)
    target = np.array(target)
    target = torch.from_numpy(target).float().cuda()
    #target = target+0.0001
    target = target.unsqueeze(1).expand(3, size, -1)
    return target

def compute_score(points, faces, target):
    score = 0
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    return torch.mean(score)

##
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)