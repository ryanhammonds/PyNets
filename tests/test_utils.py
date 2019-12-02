#!/usr/bin/env python
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import numpy as np
import os
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle
from pathlib import Path
from pynets.core import utils
import nibabel as nib
import shutil
import pytest


def test_get_file():
    base_path = utils.get_file()
    assert base_path is not None


def test_has_handle():
    f_proc = utils.has_handle(__file__)
    assert f_proc is True
    f_proc = utils.has_handle('/false/path')
    assert f_proc is False


def test_do_dir_path():
    base_dir = str(Path(__file__).parent/"examples")
    func_path = base_dir + '/002/fmri'
    func_file = func_path + '/002.nii.gz'
    atlas = 'Powers'

    if os.path.exists(func_path + '/Powers'):
        os.rmdir(func_path + '/Powers')
    dir_path = utils.do_dir_path(atlas, func_file)
    assert os.path.exists(dir_path)
    os.rmdir(dir_path)

    try:
        dir_path = utils.do_dir_path(None, func_file)
    except ValueError:
        pass


@pytest.mark.parametrize("node_size, parc", [(None, True), (6, True)])
def test_create_paths(node_size, parc):
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = 'Default'
    ID = '002'
    conn_model = 'corr'
    roi = None
    smooth = 6
    c_boot = 1000
    hpass = 100
    thr = 0.75
    thr_type = 'prop'

    # est path func
    shutil.rmtree(dir_path + "/graphs", ignore_errors=True)
    est_path_func = utils.create_est_path_func(ID, network, conn_model, thr,
                                               roi, dir_path, node_size,
                                               smooth, c_boot, thr_type, hpass,
                                               parc)
    assert est_path_func is not None
    # raw path func
    shutil.rmtree(dir_path + "/graphs", ignore_errors=True)
    raw_path_func = utils.create_raw_path_func(ID, network, conn_model, roi,
                                               dir_path, node_size, smooth,
                                               c_boot, hpass, parc)
    assert raw_path_func is not None

    # est path diffusion
    dir_path = base_dir + '/002/dmri'
    shutil.rmtree(dir_path + "/graphs", ignore_errors=True)
    target_samples = 100
    track_type = 'local'
    est_path_diff = utils.create_est_path_diff(ID, network, conn_model, thr,
                                               roi, dir_path, node_size,
                                               target_samples, track_type,
                                               thr_type, parc)
    assert est_path_diff is not None
    # raw path diffusion
    shutil.rmtree(dir_path + "/graphs", ignore_errors=True)
    raw_path_diff = utils.create_raw_path_diff(ID, network, conn_model, roi,
                                               dir_path, node_size,
                                               target_samples, track_type,
                                               parc)

    assert raw_path_diff is not None
    # graph metric path
    shutil.rmtree(dir_path + '/netmetrics', ignore_errors=True)
    graph_out_path = utils.create_csv_path(dir_path + '/graphs', est_path_diff)
    assert graph_out_path is not None

    return est_path_func, est_path_diff


@pytest.mark.parametrize("fmt",
    [
        'edgelist_csv', 'gpickle', 'graphml', 'txt', 'npy', 'edgelist_ssv',
        pytest.param(None, marks=pytest.mark.xfail)
    ]
)
def test_save_mat(fmt):
    conn_mat = np.random.rand(10, 10)
    base_dir = str(Path(__file__).parent/"examples")
    func_path = base_dir + '/002/fmri'
    struct_path = base_dir + '/002/dmri'

    if not os.path.exists(func_path + '/graphs'):
        os.mkdir(func_path + '/graphs')
    if not os.path.exists(struct_path + '/graphs'):
        os.mkdir(func_path + '/graphs')

    est_path_func = func_path + "/graphs/002_Default_est-corr_thr-0.75prop_sp"\
                                "heres-6mm_boot-1000iter_smooth-6fwhm_hpass-1"\
                                "00Hz_func.npy"
    utils.save_mat(conn_mat, est_path_func, fmt)


def test_pass_meta_io():
    # Generate paths to write mats to.
    def create_paths(thr_val):
        base_dir = str(Path(__file__).parent/"examples")
        dir_path_func = base_dir + '/002/fmri'
        dir_path_dmri = base_dir + '/002/dmri'
        network = 'Default'
        ID = '002'
        conn_model_func = 'corr'
        conn_model_dmri = 'csa'
        roi = None
        smooth = 6
        c_boot = 1000
        hpass = 100
        thr = thr_val
        thr_type = 'prop'
        parc = True
        node_size = 6
        target_samples = 100
        track_type = 'local'
        func_path = utils.create_est_path_func(ID, network, conn_model_func,
                                               thr, roi, dir_path_func,
                                               node_size, smooth, c_boot,
                                               thr_type, hpass, parc)

        struct_path = utils.create_est_path_diff(ID, network, conn_model_dmri,
                                                 thr, roi, dir_path_dmri,
                                                 node_size, target_samples,
                                                 track_type, thr_type, parc)
        return struct_path, func_path

    conn_model_func = 'cov'
    conn_model_struct = 'corr'
    network_func = 'Default'
    network_struct = "Salience"
    prune = True
    ID = '002'
    thr = 0.50
    norm = 1
    binary = True
    roi = str(Path(__file__).parent.parent/'SMALLREF2mm.nii.gz')
    est_func_path, est_struct_path = create_paths(0.5)

    # Generate npy files using names from fixtures.
    conn_mat = np.random.rand(10, 10)
    utils.save_mat(conn_mat, est_func_path)
    utils.save_mat(conn_mat, est_struct_path)
    # Test pass_meta_ins
    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist,
     binary_iterlist] = \
        utils.pass_meta_ins(conn_model_func, est_func_path, network_func,
                            thr, prune, ID, roi, norm,
                            binary)

    # Test pass_meta_ins_multi
    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist,
     binary_iterlist] = \
        utils.pass_meta_ins_multi(conn_model_func, est_func_path, network_func,
                                  thr, prune, ID, roi,
                                  norm, binary, conn_model_struct,
                                  est_struct_path, network_struct, thr,
                                  prune, ID, roi,
                                  norm, binary)

    # Test pass_meta_outs
    est_func_path_low, est_struct_path_low = create_paths(0.25)
    est_func_path_high, est_struct_path_high = create_paths(0.75)

    # Generate npy files using names from fixtures.
    conn_mat_low = np.random.rand(10, 10)
    conn_mat_high = np.random.rand(10, 10)
    utils.save_mat(conn_mat_low, est_func_path_low)
    utils.save_mat(conn_mat_high, est_func_path_high)
    est_path_iterlist = [est_func_path_low, est_func_path_high]
    thr_iterlist = [0.25, 0.50, 0.75]
    conn_model_iterlist = ['corr', 'corr', 'corr']
    network_iterlist = ['Default', 'Default', 'Default']
    prune_iterlist = [False, False, False]
    ID_iterlist = ['002', '002', '002']
    roi_iterlist = [None, None, None]
    norm_iterlist = [1, 1, 1]
    binary_iterlist = [False, False, False]
    embed = 'omni'
    multimodal = True
    multiplex = 1
    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist,
     binary_iterlist] = \
        utils.pass_meta_outs(conn_model_iterlist, est_path_iterlist,
                             network_iterlist, thr_iterlist, prune_iterlist,
                             ID_iterlist, roi_iterlist, norm_iterlist,
                             binary_iterlist, embed, multimodal, multiplex)

    '''
    # Generate npy files using names from fixtures.
    conn_mat_func_low = np.random.rand(10, 10)
    conn_mat_struct_low = np.random.rand(10, 10)
    utils.save_mat(conn_mat_func_low, est_func_path_low)
    utils.save_mat(conn_mat_struct_low, est_struct_path_low)
    est_path_iterlist = [est_func_path_low, est_struct_path_low]
    thr_iterlist = [0.25, 0.25]
    conn_model_iterlist = ['corr', 'csa']
    network_iterlist = ['Default', 'Default']
    prune_iterlist = [False, False]
    ID_iterlist = ['002', '002']
    roi_iterlist = [None, None]
    norm_iterlist = [1, 1]
    binary_iterlist = [False, False]
    embed = 'omni'
    multimodal = True
    multiplex = 1
    [conn_model_iterlist, est_path_iterlist, network_iterlist, thr_iterlist,
     prune_iterlist, ID_iterlist, roi_iterlist, norm_iterlist,
     binary_iterlist] = \
        utils.pass_meta_outs(conn_model_iterlist, est_path_iterlist,
                             network_iterlist, thr_iterlist, prune_iterlist,
                             ID_iterlist, roi_iterlist, norm_iterlist,
                             binary_iterlist, embed, multimodal, multiplex)
    '''

'''
def test_save_RSN_coords_and_labels_to_pickle():
    """
    Test save_RSN_coords_and_labels_to_pickle functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    coord_file_path = dir_path + '/DesikanKlein2012/Default_coords_rsn.pkl'
    coord_file = open(coord_file_path, 'rb')
    coords = pickle.load(coord_file)
    labels_file_path = dir_path + '/DesikanKlein2012/Default_labels_rsn.pkl'
    labels_file = open(labels_file_path, 'rb')
    labels = pickle.load(labels_file)
    network = None

    [coord_path, labels_path] = utils.save_RSN_coords_and_labels_to_pickle(coords, labels, dir_path, network)
    assert os.path.isfile(coord_path) is True
    assert os.path.isfile(labels_path) is True


def test_save_nifti_parcels_map():
    """
    Test save_nifti_parcels_map functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    ID = '002'
    dir_path = base_dir + '/002/fmri'
    roi = None
    network = None
    array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
    affine = np.diag([1, 2, 3, 1])
    net_parcels_map_nifti = nib.Nifti1Image(array_data, affine)

    net_parcels_nii_path = utils.save_nifti_parcels_map(ID, dir_path, roi, network, net_parcels_map_nifti)
    assert os.path.isfile(net_parcels_nii_path) is True


def test_save_ts_to_file():
    """
    Test save_ts_to_file functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    roi = None
    c_boot = 3
    smooth = 2
    hpass = None
    network = None
    node_size = 'parc'
    ID = '002'
    dir_path = base_dir + '/002/fmri'
    ts_within_nodes = '/tmp/'
    out_path_ts = utils.save_ts_to_file(roi, network, ID, dir_path, ts_within_nodes, c_boot, smooth, hpass, node_size)
    assert os.path.isfile(out_path_ts) is True


# def test_build_embedded_connectome():
#     """
#     Test build_embedded_connectome functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     ID = '002'
#     multimodal = False
#     types = ['omni', 'mase']
#     est_path_iterlist = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
#                          dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
#     for type in types:
#         out_path = utils.build_embedded_connectome(est_path_iterlist, ID, multimodal, type)
#         assert out_path is not None


def test_check_est_path_existence():
    """
    Test check_est_path_existence functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    est_path_list = [dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.09dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.08dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.07dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.06dens_100000samples_particle_track.npy',
                     dir_path + '/DesikanKlein2012/0021001_Default_est_csd_0.05dens_100000samples_particle_track.npy']
    [est_path_list_ex, _] = utils.check_est_path_existence(est_path_list)
    assert est_path_list_ex is not None


# def test_collect_pandas_df():
#     """
#     Test collect_pandas_df functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     multi_nets = ['Default', 'SalVentAttn']
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     multimodal = False
#     net_mets_csv_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.1_parc.csv']
#     utils.collect_pandas_df(network, ID, net_mets_csv_list, plot_switch, multi_nets, multimodal)
#
#
# def test_collect_pandas_df_make():
#     """
#     Test collect_pandas_df_make functionality
#     """
#     base_dir = str(Path(__file__).parent/"examples")
#     dir_path = base_dir + '/002/dmri'
#     network = 'Default'
#     ID = '002'
#     plot_switch = True
#     net_pickle_mt_list = [dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csa_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_csd_thr-0.1_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.05_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.06_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.07_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.08_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.09_parc.csv',
#                           dir_path + '/DesikanKlein2012/0021001_net_mets_Default_tensor_thr-0.1_parc.csv']
#     utils.collect_pandas_df_make(net_pickle_mt_list, ID, network, plot_switch)


def test_create_est_path_func():
    """
    Test create_est_path_diff functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    network = 'Default'
    ID = '002'
    models = ['corr', 'cov', 'sps', 'partcorr']
    roi = None
    node_size = 6
    smooth = 6
    c_boot = 1000
    hpass = 100
    parc = True

    # Cross test various connectivity models, thresholds, and parc true/false.
    for conn_model in models:
        for val in range(1, 10):
            thr = round(val*0.1, 1)
            for thr_type in ['prop', 'abs', 'dens', 'mst', 'disp']:
                for parc in [True, False]:
                    est_path = utils.create_est_path_func(ID, network, conn_model, thr, roi, dir_path, node_size,
                                                          smooth, c_boot,
                                               thr_type, hpass, parc)
                    assert est_path is not None


def test_flatten():
    """
    Test list flatten functionality
    """
    # Slow, but successfully flattens a large array
    l = np.random.rand(3, 3, 3).tolist()
    l = utils.flatten(l)

    i = 0
    for item in l:
        i += 1
    assert i == (3*3*3)


def test_merge_dicts():
    """
    Test merge_dicts functionality
    """
    x = {
        'a': 1,
        'b': 2,
        'c': 3
    }
    y = {
        'd': 4,
        'e': 5,
        'f': 6
    }
    z = utils.merge_dicts(x, y)

    dic_len = len(x)+len(y)
    assert len(z) == dic_len
'''
