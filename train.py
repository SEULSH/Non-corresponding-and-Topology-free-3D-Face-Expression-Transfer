import os
import torch
from tqdm import tqdm
import numpy as np
import pickle
import trimesh
from torch.utils.data import DataLoader
from psbody.mesh import Mesh

import utils
from utils import get_adj, sparse_mx_to_torch_sparse_tensor
from data import ShapeData, COMA_DATA
from model_for_GT import PaiAutoencoder
from model import ExpressionTransfer

##
device = torch.device("cuda")
GPU = True
device_idx = 0
torch.cuda.get_device_name(device_idx)

meshpackage = 'mpi-mesh'
root_dir = '/home/lsh/lsh_data/'
dataname = 'COMA_data'
datapath = root_dir + dataname
template_path = root_dir + dataname+'/template'

### build autoencoder model and load pretrained parameters for generating GT expression transferred shapes
latent_size = 512
generative_model = 'tiny-conv'
downsample_method = 'COMA_downsample'
reference_mesh_file = os.path.join(template_path, 'template.obj')
downsample_directory = os.path.join(template_path, downsample_method)
ds_factors = [4, 4, 4, 4]
kernal_size = [9, 9, 9, 9, 9]
step_sizes = [2, 2, 1, 1, 1]
filter_sizes_enc = [3, 16, 32, 64, 128]
filter_sizes_dec = [128, 64, 32, 32, 16, 3]
args = {'generative_model': generative_model,
        'datapath': datapath,
        'results_folder': os.path.join(root_dir + dataname, 'results/' + generative_model),
        'reference_mesh_file': reference_mesh_file, 'downsample_directory': downsample_directory,
        'checkpoint_file': 'checkpoint',
        'seed': 2, 'loss': 'l1',
        'batch_size': 32, 'num_epochs': 300, 'eval_frequency': 200, 'num_workers': 4,
        'filter_sizes_enc': filter_sizes_enc, 'filter_sizes_dec': filter_sizes_dec,
        'nz': latent_size,
        'ds_factors': ds_factors, 'step_sizes': step_sizes,
        'lr': 1e-3, 'regularization': 5e-5,
        'scheduler': True, 'decay_rate': 0.99, 'decay_steps': 1,
        'resume': True,
        'mode': 'test', 'shuffle': True, 'nVal': 100, 'normalization': True}

np.random.seed(args['seed'])
print("Loading data .. ")
shapedata = ShapeData(datapath=args['datapath'],
                       normalization=args['normalization'],
                       template_file=args['reference_mesh_file'],
                       meshpackage=meshpackage)
print("Loading Transform Matrices ..")
with open(os.path.join(args['downsample_directory'], 'downsampling_matrices.pkl'), 'rb') as fp:
    downsampling_matrices = pickle.load(fp)
M_verts_faces = downsampling_matrices['M_verts_faces']
if shapedata.meshpackage == 'mpi-mesh':
    M = [Mesh(v=M_verts_faces[i][0], f=M_verts_faces[i][1]) for i in range(len(M_verts_faces))]
elif shapedata.meshpackage == 'trimesh':
    M = [trimesh.base.Trimesh(vertices=M_verts_faces[i][0], faces=M_verts_faces[i][1], process=False) for i
         in range(len(M_verts_faces))]
A = downsampling_matrices['A']
D = downsampling_matrices['D']
U = downsampling_matrices['U']
# add zero last points for each level template
vertices = [torch.cat(
    [torch.tensor(M_verts_faces[i][0], dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) for i in range(len(M_verts_faces))]
#
if shapedata.meshpackage == 'mpi-mesh':
    sizes = [x.v.shape[0] for x in M]
elif shapedata.meshpackage == 'trimesh':
    sizes = [x.vertices.shape[0] for x in M]
print("Loading adj Matrices ..")
with open(os.path.join(args['downsample_directory'], 'pai_matrices.pkl'), 'rb') as fp:
    [Adj, sizes, bD, bU] = pickle.load(fp)

tD = [sparse_mx_to_torch_sparse_tensor(s) for s in bD]
tU = [sparse_mx_to_torch_sparse_tensor(s) for s in bU]

model = PaiAutoencoder(filters_enc=args['filter_sizes_enc'],
                       filters_dec=args['filter_sizes_dec'],
                       latent_size=args['nz'],
                       sizes=sizes,
                       t_vertices=vertices,  # template vertex after add last zero nodes
                       num_neighbors=kernal_size,
                       x_neighbors=Adj,
                       D=tD, U=tU, flag=2).to(device)

checkpoint_path = os.path.join(args['results_folder'], 'latent_'+str(latent_size), 'checkpoints')
checkpoint_dict = torch.load(os.path.join(checkpoint_path, args['checkpoint_file'] + '.pth.tar'),
                             map_location=device)
model_dict = model.state_dict()
pretrained_dict = checkpoint_dict['autoencoder_state_dict']
if next(iter(pretrained_dict)).startswith("module."):
    for k, v in pretrained_dict.items():
        name = k[7:]  # remove `module.`
        model_dict[name] = v
model.load_state_dict(model_dict)
model.to("cpu")
model.eval()

########### following is our dataset, network and training process #####################
print('training on COMA dataset')
dataset = COMA_DATA(datapath=datapath, mode='train', model=model, shapedata=shapedata)

save_root = './saved_model' + '/' + dataname
if not os.path.exists(save_root):
    os.makedirs(save_root)

batch_size = 10
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

our_model = ExpressionTransfer()
our_model.cuda()
total = sum(p.numel() for p in our_model.parameters())
print("Total params of our model: %.2fM" % (total / 1e6))

old_lr = initial_lr = 0.0001
optimizer_G = torch.optim.Adam(our_model.parameters(), lr=initial_lr, betas=(0, 0.9))

# checkpoint = torch.load(save_root+'/199.model', map_location=device)
# our_model.load_state_dict(checkpoint)
# our_model.eval()

best_loss = 1000000.0
for epoch in range(0, 200):

    if epoch > 100:
        lrd = initial_lr / 100
        new_lr = initial_lr - (epoch-100)*lrd
    else:
        new_lr = old_lr

    if new_lr != old_lr:
        new_lr_G = new_lr

        for param_group in optimizer_G.param_groups:
            param_group['lr'] = new_lr_G
        print('update learning rate: %f -> %f' % (old_lr, new_lr))
        old_lr = new_lr

    total_loss = 0
    total_rec_loss = 0
    total_edge_loss = 0
    for j, data in enumerate(tqdm(dataloader)):

        optimizer_G.zero_grad()

        target_shape, source_shape, GT_shape, new_face_T = data
        target_shape = target_shape.to(device)
        source_shape = source_shape.to(device)
        GT_shape = GT_shape.to(device)

        deformed_shape = our_model(source_shape, target_shape)

        r_loss = torch.mean((deformed_shape - GT_shape)**2)

        # edg_loss = 0
        # for i in range(deformed_shape.shape[0]):  #
        #     f = new_face_T[i].cpu().numpy()
        #     v = target_shape[i].cpu().numpy()
        #     edg_loss = edg_loss + utils.compute_score(deformed_shape[i].unsqueeze(0), f, utils.get_target(v, f, 1))
        # edg_loss = edg_loss / deformed_shape.shape[0]
        #
        # loss = 1000.0 * r_loss + 0.5 * edg_loss
        loss = 1000.0 * r_loss

        loss.backward()
        optimizer_G.step()

        total_loss = total_loss + loss.item()
        total_rec_loss = total_rec_loss + 1000.0 * r_loss.item()
        # total_edge_loss = total_edge_loss + 0.5 * edg_loss.item()

    print('####################################')
    print(epoch)
    mean_loss = float((total_loss / (j + 1)))
    mean_rec_loss = float((total_rec_loss / (j + 1)))
    # mean_edg_loss = float((total_edge_loss / (j + 1)))

    print('epoch: ', epoch, 'mean_loss: ', mean_loss)
    print('epoch: ', epoch, 'mean_rec_loss: ', mean_rec_loss)
    # print('epoch: ', epoch, 'mean_edg_loss: ', mean_edg_loss)
    print('####################################')

    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(our_model.state_dict(), save_root + '/best.model')
    if (epoch + 1) % 5 == 0:
        save_path = save_root + '/' + str(epoch) + '.model'
        torch.save(our_model.state_dict(), save_path)