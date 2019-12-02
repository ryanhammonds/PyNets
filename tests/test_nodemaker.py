#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017

@authors: Derek Pisner & Ryan Hammonds

"""
import os.path as op
import pytest
import numpy as np
import time
import nibabel as nib
from pathlib import Path
from pynets.core import nodemaker
try:
    import cPickle as pickle
except ImportError:
    import _pickle as pickle


def test_get_sphere():
    """
    Test get_sphere functionality
    """
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    img_file = dir_path + '/nodif_b0_bet.nii.gz'
    img = nib.load(img_file)
    r = 4
    vox_dims = (2.0, 2.0, 2.0)
    coords_file = dir_path + '/DesikanKlein2012/Default_coords_rsn.pkl'
    with open(coords_file, 'rb') as file_:
        coords = pickle.load(file_)
    neighbors = []
    for coord in coords:
        neighbors.append(nodemaker.get_sphere(coord, r, vox_dims, img.shape[0:3]))
    neighbors = [i for i in neighbors if len(i)>0]
    assert len(neighbors) == 4
    

@pytest.mark.parametrize("atlas,parc",
    [
        ('atlas_aal', False),
        ('atlas_talairach_gyrus', False),
        ('atlas_talairach_ba', False),
        ('atlas_talairach_lobe', False),
        ('atlas_harvard_oxford', False),
        ('atlas_destrieux_2009', False),
        ('atlas_pauli_2017', True),
        ('atlas_pauli_2017', False),
        ('coords_dosenbach_2010', False),
        ('atlas_msdl', False),
        (pytest.param('atlas_yeo_2011', False, marks=pytest.mark.xfail(raises=ValueError))),
        (pytest.param('atlas_allen_2011', False, marks=pytest.mark.xfail(raises=ValueError)))
    ]
)
def test_nilearn_atlas_helper(atlas, parc):
    [labels, networks_list, parlistfile] = nodemaker.nilearn_atlas_helper(atlas, parc)
    assert labels is not None


@pytest.mark.parametrize('network,file_exists',
    [
        ('Default', True),
        ('DefaultA', True),
        pytest.param('Default', False, marks=pytest.mark.xfail(raises=ValueError)),
        pytest.param('Invalid', True, marks=pytest.mark.xfail(raises=ValueError))
    ]
)
def test_nodemaker_tools_parlistfile_RSN(network, file_exists):
    """
    Test nodemaker_tools_parlistfile_RSN functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    atlas_dir = str(Path(__file__).parent.parent/"pynets/core/atlases")
    func_file = dir_path + '/002.nii.gz'

    if not file_exists:
        parlistfile = base_dir + '/false_file.nii.gz'
        try:
            nodemaker.get_names_and_coords_of_parcels(parlistfile)
        except ValueError:
            pass
        nodemaker.get_names_and_coords_of_parcels(parlistfile)
    else:
        parlistfile = atlas_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    parc = True

    start_time = time.time()
    [coords, _, _] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))
    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    start_time = time.time()
    parcel_list = nodemaker.gen_img_list(parlistfile)
    [net_coords, net_parcel_list, net_labels, network] = \
        nodemaker.get_node_membership(network, func_file, coords, labels, parc,
                                      parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list)
    print("%s%s%s" % ('create_parcel_atlas --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_labels, dir_path)
    print("%s%s%s" % ('gen_network_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_labels is not None
    assert net_parcel_list is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert network is not None


@pytest.mark.parametrize("atlas,network,vox_dim",
    [
        ('coords_dosenbach_2010', 'Default', '2mm'),
        pytest.param('coords_dosenbach_2010', 'Default', '1mm',
                     marks=pytest.mark.xfail(raises=ValueError)),
        ('coords_dosenbach_2010', 'DefaultA', '2mm'),
        pytest.param('coords_dosenbach_2010', 'DefaultA', '1mm',
                     marks=pytest.mark.xfail(raises=ValueError)),
        ('coords_power_2011', 'Default', '2mm'),
        pytest.param('atlas_yeo_2011', 'Default', '2mm',
                     marks=pytest.mark.xfail(raises=AttributeError))
    ]
)
def test_nodemaker_tools_nilearn_coords_RSN(atlas, network, vox_dim):
    """
    Test nodemaker_tools_nilearn_coords_RSN functionality

    Note: Nilearn coords will fail when '1mm' is selected since the
          coords are still in 2mm space. '1mm' artificially decreases
          voxels size without resampling for faster computation.
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    parc = False
    parcel_list = None
    func_file = dir_path + '/002.nii.gz'
    if vox_dim == '1mm' and not op.isfile(dir_path + '/002_1mm.nii.gz'):
        img_data = nib.load(func_file).get_data()
        img_highres = nib.Nifti1Image(img_data, np.eye(4))
        img_highres.to_filename(dir_path + '/002_1mm.nii.gz')
        func_file = dir_path + '/002_1mm.nii.gz'
    elif vox_dim == '1mm':
        func_file = dir_path + '/002_1mm.nii.gz'

    start_time = time.time()
    [coords, _, _, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords, _, net_labels, network] = nodemaker.get_node_membership(network, func_file, coords, labels, parc,
                                                                         parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert labels is not None
    assert net_coords is not None
    assert net_labels is not None
    assert network is not None


def test_nodemaker_tools_masking_parlistfile_RSN():
    """
    Test nodemaker_tools_masking_parlistfile_RSN functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    roi = base_dir + '/pDMN_3_bin.nii.gz'
    network = 'Default'
    ID = '002'
    perc_overlap = 0.10
    parc = True

    start_time = time.time()
    [coords, _, _] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    labels = np.arange(len(coords) + 1)[np.arange(len(coords) + 1) != 0].tolist()

    start_time = time.time()
    parcel_list = nodemaker.gen_img_list(parlistfile)
    [net_coords, net_parcel_list, net_labels, network] = nodemaker.get_node_membership(network, func_file, coords,
                                                                                            labels, parc,
                                                                                            parcel_list)
    print("%s%s%s" % ('get_node_membership --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords_masked, net_labels_masked, net_parcel_list_masked] = nodemaker.parcel_masker(roi, net_coords,
                                                                                                  net_parcel_list,
                                                                                                  net_labels,
                                                                                                  dir_path, ID,
                                                                                                  perc_overlap)
    print("%s%s%s" % ('parcel_masker --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(net_parcel_list_masked)
    print("%s%s%s" % ('create_parcel_atlas --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    out_path = nodemaker.gen_network_parcels(parlistfile, network, net_labels_masked, dir_path)
    print("%s%s%s" % ('gen_network_parcels --> finished: ', str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_labels is not None
    assert net_parcel_list is not None
    assert net_coords_masked is not None
    assert net_labels_masked is not None
    assert net_parcel_list_masked is not None
    assert out_path is not None
    assert net_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert network is not None


@pytest.mark.parametrize("atlas", ['coords_dosenbach_2010', 'coords_power_2011'])
def test_nodemaker_tools_masking_coords_RSN(atlas):
    """
    Test nodemaker_tools_masking_coords_RSN functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path= base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    roi = base_dir + '/pDMN_3_bin.nii.gz'
    network = 'Default'
    parc = False
    parcel_list = None
    error = 2
    start_time = time.time()
    [coords, _, _, labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Masking RSN version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords, _, net_labels, network] = nodemaker.get_node_membership(network, func_file, coords, labels, parc,
                                                                         parcel_list)
    print("%s%s%s" % ('get_node_membership (Masking RSN version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [net_coords_masked, net_labels_masked] = nodemaker.coords_masker(roi, net_coords, net_labels, error)
    print("%s%s%s" % ('coords_masker (Masking RSN version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert coords is not None
    assert net_coords is not None
    assert net_coords_masked is not None
    assert net_labels is not None
    assert net_labels_masked is not None
    assert network is not None


def test_nodemaker_tools_parlistfile_WB():
    """
    Test nodemaker_tools_parlistfile_WB functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'

    start_time = time.time()
    [WB_coords, _, _] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels (User-atlas whole-brain version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    WB_labels = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()

    start_time = time.time()

    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list)
    print("%s%s%s" % ('create_parcel_atlas (User-atlas whole-brain version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_labels is not None
    assert WB_parcel_list is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None


@pytest.mark.parametrize("atlas", ['coords_dosenbach_2010', 'coords_power_2011'])
def test_nodemaker_tools_nilearn_coords_WB(atlas):
    """
    Test nodemaker_tools_nilearn_coords_WB functionality
    """
    start_time = time.time()
    [WB_coords, _, _, WB_labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Whole-brain version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_labels is not None


def test_nodemaker_tools_masking_parlistfile_WB():
    """
    Test nodemaker_tools_masking_parlistfile_WB functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas = 'whole_brain_cluster_labels_PCA200'
    roi = base_dir + '/pDMN_3_bin.nii.gz'
    mask = None
    ID = '002'
    parc = True
    perc_overlap = 0.10

    start_time = time.time()
    [WB_coords, _, _] = nodemaker.get_names_and_coords_of_parcels(parlistfile)
    print("%s%s%s" % ('get_names_and_coords_of_parcels (Masking whole-brain version) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    WB_labels = np.arange(len(WB_coords) + 1)[np.arange(len(WB_coords) + 1) != 0].tolist()

    start_time = time.time()
    WB_parcel_list = nodemaker.gen_img_list(parlistfile)
    [_, _, WB_parcel_list_masked] = nodemaker.parcel_masker(roi, WB_coords, WB_parcel_list, WB_labels, dir_path,
                                                            ID, perc_overlap)
    print("%s%s%s" % ('parcel_masker (Masking whole-brain version) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))

    start_time = time.time()
    [WB_parcels_map_nifti, parcel_list_exp] = nodemaker.create_parcel_atlas(WB_parcel_list_masked)
    print("%s%s%s" % ('create_parcel_atlas (Masking whole-brain version) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))

    start_time = time.time()
    [WB_net_parcels_map_nifti_unmasked, WB_coords_unmasked, _,
     WB_atlas, WB_uatlas, dir_path] = nodemaker.node_gen(WB_coords, WB_parcel_list, WB_labels,
                                                                        dir_path, ID, parc, atlas, parlistfile)
    print("%s%s%s" % ('node_gen (Masking whole-brain version) --> finished: ',
    np.round(time.time() - start_time, 1), 's'))

    start_time = time.time()
    [WB_net_parcels_map_nifti_masked, WB_coords_masked, WB_labels_masked,
     WB_atlas, WB_uatlas, dir_path] = nodemaker.node_gen_masking(roi, WB_coords, WB_parcel_list,
                                                                                WB_labels, dir_path, ID, parc,
                                                                                atlas, parlistfile)

    print("%s%s%s" % ('node_gen_masking (Masking whole-brain version) --> finished: ',
                      np.round(time.time() - start_time, 1), 's'))

    assert WB_coords is not None
    assert WB_labels is not None
    assert WB_parcel_list is not None
    assert WB_coords_masked is not None
    assert WB_labels_masked is not None
    assert WB_parcel_list_masked is not None
    assert WB_parcels_map_nifti is not None
    assert parcel_list_exp is not None
    assert WB_net_parcels_map_nifti_unmasked is not None
    assert WB_coords_unmasked is not None
    assert WB_net_parcels_map_nifti_masked is not None
    assert WB_coords_masked is not None


@pytest.mark.parametrize("atlas", ['coords_dosenbach_2010', 'coords_power_2011'])
def test_nodemaker_tools_masking_coords_WB(atlas):
    """
    Test nodemaker_tools_masking_coords_WB functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    roi = base_dir + '/pDMN_3_bin.nii.gz'
    error = 2

    start_time = time.time()
    [WB_coords, _, _, WB_labels] = nodemaker.fetch_nilearn_atlas_coords(atlas)
    print("%s%s%s" % ('fetch_nilearn_atlas_coords (Masking whole-brain coords version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    start_time = time.time()
    [WB_coords_masked, WB_labels_masked] = nodemaker.coords_masker(roi, WB_coords, WB_labels, error)
    print("%s%s%s" % ('coords_masker (Masking whole-brain coords version) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert WB_coords is not None
    assert WB_coords is not None
    assert WB_coords_masked is not None
    assert WB_labels is not None
    assert WB_labels_masked is not None


def test_WB_fetch_nodes_and_labels1():
    """
    Test WB_fetch_nodes_and_labels1 functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    use_AAL_naming = True
    ref_txt = None
    parc = True

    start_time = time.time()
    [_, coords, atlas_name, _, parcel_list, par_max, parlistfile,
     dir_path] = nodemaker.fetch_nodes_and_labels(atlas, parlistfile, ref_txt, parc, func_file, use_AAL_naming)
    print("%s%s%s" % ('WB_fetch_nodes_and_labels (Parcel Nodes) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert coords is not None
    assert dir_path is not None


def test_WB_fetch_nodes_and_labels2():
    """
    Test WB_fetch_nodes_and_labels2 functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    ref_txt = None
    parc = False
    use_AAL_naming = True
    start_time = time.time()
    [_, coords, atlas_name, _, _, par_max, parlistfile,
     _] = nodemaker.fetch_nodes_and_labels(atlas, parlistfile, ref_txt, parc, func_file, use_AAL_naming)
    print("%s%s%s" % ('WB_fetch_nodes_and_labels (Spherical Nodes) --> finished: ',
    str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert atlas_name is not None
    assert coords is not None


def test_RSN_fetch_nodes_and_labels1():
    """
    Test RSN_fetch_nodes_and_labels1 functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    ref_txt = None
    parc = True
    use_AAL_naming = True

    start_time = time.time()
    [RSN_labels, RSN_coords, atlas_name, _, parcel_list, par_max,
     parlistfile, _] = nodemaker.fetch_nodes_and_labels(atlas, parlistfile, ref_txt, parc,
                                                        func_file, use_AAL_naming)
    print("%s%s%s" % ('RSN_fetch_nodes_and_labels (Parcel Nodes) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert parcel_list is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_labels is not None


def test_RSN_fetch_nodes_and_labels2():
    """
    Test RSN_fetch_nodes_and_labels2 functionality
    """
    # Set example inputs
    base_dir = str(Path(__file__).parent/"examples")
    parlistfile = base_dir + '/whole_brain_cluster_labels_PCA200.nii.gz'
    atlas = 'whole_brain_cluster_labels_PCA200'
    dir_path = base_dir + '/002/fmri'
    func_file = dir_path + '/002.nii.gz'
    ref_txt = None
    parc = False
    use_AAL_naming = True

    start_time = time.time()
    [RSN_labels, RSN_coords, atlas_name, _, _,
     par_max, parlistfile, _] = nodemaker.fetch_nodes_and_labels(atlas, parlistfile, ref_txt, parc, func_file,
                                                                 use_AAL_naming)
    print("%s%s%s" % ('RSN_fetch_nodes_and_labels (Spherical Nodes) --> finished: ',
                      str(np.round(time.time() - start_time, 1)), 's'))

    assert parlistfile is not None
    assert par_max is not None
    assert atlas_name is not None
    assert RSN_coords is not None
    assert RSN_labels is not None


def test_create_spherical_roi_volumes():
    """
    Test create_spherical_roi_volumes functionality
    """
    import pkg_resources
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/dmri'
    node_size = 2
    vox_size = '2mm'
    template_mask = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size +
                                                    "_brain_mask.nii.gz")
    coords_file = dir_path + '/DesikanKlein2012/Default_coords_rsn.pkl'
    with open(coords_file, 'rb') as file_:
        coords = pickle.load(file_)
    [parcel_list, par_max, node_size, parc] = nodemaker.create_spherical_roi_volumes(node_size, coords, template_mask)
    assert len(parcel_list) > 0
    

def test_mask_roi():
    """
    Test mask_roi functionality
    """
    import pkg_resources
    vox_size = '2mm'
    mask = pkg_resources.resource_filename("pynets", "templates/MNI152_T1_" + vox_size + "_brain_mask.nii.gz")
    base_dir = str(Path(__file__).parent/"examples")
    dir_path = base_dir + '/002/fmri'
    img_file = dir_path + '/002.nii.gz'
    roi = base_dir + '/pDMN_3_bin.nii.gz'
    roi_masked = nodemaker.mask_roi(dir_path, roi, mask, img_file)
    assert roi_masked is not None


