import os
from torch.utils.data import Dataset
import torch
import numpy as np
from psbody.mesh import Mesh

def angle_axis(angle, axis):
    ## type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    ).to(torch.float32)
    # yapf: enable
    return R

class PointcloudRotatePerturbation():
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):

        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        # angles = np.clip(
        #     self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        # )

        angles = np.random.randn(3) * 2 * np.pi

        return angles

    def __call__(self):
        angles = self._get_angles()

        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))

        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))

        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        return rotation_matrix.t()

class ShapeData(object):
    def __init__(self, datapath, normalization, template_file, meshpackage):

        self.datapath = datapath

        self.normalization = normalization

        self.meshpackage = meshpackage

        self.reference_mesh = Mesh(filename=template_file)

        self.mean = np.load(datapath + '/mean.npy')
        self.std = np.load(datapath + '/std.npy')
        self.n_vertex = self.mean.shape[0]
        self.n_features = self.mean.shape[1]


class COMA_DATA(Dataset):
    def __init__(self, datapath=None, mode='train', model=None, shapedata=None):

        self.path_files = []
        self.persons = []
        self.datapath = datapath + '/' + mode
        for p in os.listdir(self.datapath):
            self.persons.append(p)
            for ex in os.listdir(self.datapath+'/'+p):
                for face in os.listdir(self.datapath+'/'+p+'/'+ex):
                    self.path_files.append(self.datapath+'/'+p+'/'+ex+'/'+face)

        self.neutral_path = datapath + '/neutral_faces'

        self.model = model
        self.shapedata = shapedata
        self.mean = torch.tensor(self.shapedata.mean, dtype=torch.float32)
        self.std = torch.tensor(self.shapedata.std, dtype=torch.float32)
        self.template = shapedata.reference_mesh

        ## for rotate shapes
        # self.rotate = PointcloudRotatePerturbation(angle_sigma=0.06, angle_clip=0.18)

    def __getitem__(self, index):
        ##
        # s_mesh = Mesh(filename=self.path_files[index])
        # s_mesh.write_obj('./transfer/'+str(index)+'.obj')
        ##

        source_E_shape = Mesh(filename=self.path_files[index]).v
        source_E_shape = torch.tensor(source_E_shape, dtype=torch.float32)

        source_E_identity = self.path_files[index].split('/')[-3]
        source_N_shape = Mesh(filename=self.neutral_path+'/'+source_E_identity+'.ply').v
        source_N_shape = torch.tensor(source_N_shape, dtype=torch.float32)

        source_E_shape0 = (source_E_shape - self.mean) / self.std
        source_N_shape0 = (source_N_shape - self.mean) / self.std
        ##

        ##
        target_id = np.random.randint(0, len(self.path_files))
        ##
        # t_mesh = Mesh(filename=self.path_files[target_id])
        # t_mesh.write_obj('./transfer/'+str(target_id)+'.obj')
        ##
        target_E_shape = Mesh(filename=self.path_files[target_id]).v
        target_E_shape = torch.tensor(target_E_shape, dtype=torch.float32)

        target_E_identity = self.path_files[target_id].split('/')[-3]
        target_N_shape = Mesh(filename=self.neutral_path+'/'+target_E_identity+'.ply').v
        target_N_shape = torch.tensor(target_N_shape, dtype=torch.float32)
        target_N_shape0 = (target_N_shape - self.mean) / self.std
        ##
        with torch.no_grad():
            generated_GT_shape = self.model(None, source_N_shape0, source_E_shape0, target_N_shape0)[0:-1]

        ##
        generated_GT_shape = generated_GT_shape * self.std + self.mean

        # g_mesh = Mesh(v=generated_GT_shape, f=t_mesh.f)
        # g_mesh.write_obj('./transfer/'+str(index)+'_'+str(target_id)+'_g.obj')

        ## normalize
        target_E_shape = target_E_shape - torch.mean(target_E_shape, dim=0)
        # target_E_shape = target_E_shape / torch.max(torch.sqrt(torch.sum(target_E_shape**2, dim=-1)))

        generated_GT_shape = generated_GT_shape - torch.mean(generated_GT_shape, dim=0)
        # generated_GT_shape = generated_GT_shape / torch.max(torch.sqrt(torch.sum(generated_GT_shape**2, dim=-1)))

        source_E_shape = source_E_shape - torch.mean(source_E_shape, dim=0)
        # source_E_shape = source_E_shape / torch.max(torch.sqrt(torch.sum(source_E_shape**2, dim=-1)))

        # ## rotate 3 shapes randomly
        # t_matrix = self.rotate.__call__()
        # target_E_shape = torch.matmul(target_E_shape, t_matrix)
        # # t_mesh = Mesh(v=target_E_shape, f=self.template.f)
        # # t_mesh.write_obj('./transfer/' + str(target_id) + '_t.obj')
        # generated_GT_shape = torch.matmul(generated_GT_shape, t_matrix)
        # # t_mesh = Mesh(v=generated_GT_shape, f=self.template.f)
        # # t_mesh.write_obj('./transfer/' + str(target_id) + '_g.obj')
        # s_matrix = self.rotate.__call__()
        # source_E_shape = torch.matmul(source_E_shape, s_matrix)
        # # t_mesh = Mesh(v=source_E_shape, f=self.template.f)
        # # t_mesh.write_obj('./transfer/' + str(target_id) + '_' + str(index) + '_s.obj')

        # # shuffle orders
        # template_face = self.template.f

        npoints = generated_GT_shape.shape[0]
        random_sample = np.random.choice(npoints, size=npoints, replace=False)
        random_sample2 = np.random.choice(npoints, size=npoints, replace=False)
        target_E_shape = target_E_shape[random_sample2]
        source_E_shape = source_E_shape[random_sample]
        generated_GT_shape = generated_GT_shape[random_sample2]

        # ## get new faces
        # face_dict = {}
        # for i in range(len(random_sample)):
        #     face_dict[random_sample[i]] = i
        # new_f = []
        # for i in range(len(template_face)):
        #     new_f.append(
        #         [face_dict[template_face[i][0]], face_dict[template_face[i][1]], face_dict[template_face[i][2]]])
        # new_face_S = np.array(new_f)
        # ## get new faces
        # face_dict = {}
        # for i in range(len(random_sample2)):
        #     face_dict[random_sample2[i]] = i
        # new_f = []
        # for i in range(len(template_face)):
        #     new_f.append(
        #         [face_dict[template_face[i][0]], face_dict[template_face[i][1]], face_dict[template_face[i][2]]])
        # new_face_T = np.array(new_f)
        #
        # # return target_E_shape, source_E_shape, generated_GT_shape  #, new_face_S, new_face_T
        # return target_E_shape, source_E_shape, generated_GT_shape, new_face_T
        new_face_T = []
        return target_E_shape, source_E_shape, generated_GT_shape, new_face_T

    def __len__(self):
        return len(self.path_files)


if __name__ == '__main__':

    import pickle
    import trimesh
    from utils import get_adj, sparse_mx_to_torch_sparse_tensor
    from model_for_GT import PaiAutoencoder
    from psbody.mesh import Mesh

    ##
    device = torch.device("cuda")
    GPU = True
    device_idx = 0
    torch.cuda.get_device_name(device_idx)

    meshpackage = 'mpi-mesh'
    root_dir = '/home/lsh/lsh_data/'
    dataname = 'COMA_data'
    datapath = root_dir + dataname
    template_path = root_dir + dataname + '/template'

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
        [torch.tensor(M_verts_faces[i][0], dtype=torch.float32), torch.zeros((1, 3), dtype=torch.float32)], 0).to(device) for i in
        range(len(M_verts_faces))]

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

    checkpoint_path = os.path.join(args['results_folder'], 'latent_' + str(latent_size), 'checkpoints')
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
    ###################################
    test_data = COMA_DATA(datapath, mode='train', model=model, shapedata=shapedata)
    a,b,c,d = test_data.__getitem__(10151)
    print('')

