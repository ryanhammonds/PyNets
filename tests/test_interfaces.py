#!/usr/bin/env python

import os
from shutil import copyfile
import tempfile
from pathlib import Path
import pytest

import numpy as np
import pandas as pd
import nibabel as nib
import nipype.pipeline.engine as pe

import pynets.core.interfaces as interfaces

@pytest.mark.parametrize("atlas, uatlas, AAL", [
    ('atlas_harvard_oxford', None, False),
    pytest.param(None, None, False, marks=pytest.mark.xfail)
])
def test_FetchNodesLabels(atlas, uatlas, AAL):

    root_dir =  str(Path(__file__).parent.parent)
    base_dir = str(Path(__file__).parent/"examples")

    temp_dir = tempfile.TemporaryDirectory()
    outdir = str(temp_dir.name)

    node = pe.Node(interfaces.FetchNodesLabels(), name='test')
    node.inputs.atlas = atlas
    node.inputs.uatlas = uatlas
    node.inputs.parc = False
    node.inputs.in_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-T1w_desc-preproc_bold.nii.gz"
    node.inputs.outdir = outdir
    node.inputs.use_AAL_naming = AAL

    if uatlas is True:
        uatlas = f"{root_dir}/pynets/core/atlases/Harvos.path.copye/labelcharts/HarvardOxfordThr252mmWholeBrainMakris2006.txt"
        node.inputs.clustering = True
    else:
        node.inputs.ref_txt = None

    out = node.run()
    assert out is not None

    temp_dir.cleanup()



def test_NetworkAnalysis():

    temp_dir = tempfile.TemporaryDirectory()
    base_dir = str(Path(__file__).parent/"examples")
    outdir = str(temp_dir.name) + '/miscellaneous'
    os.mkdir(outdir)

    node = pe.Node(interfaces.NetworkAnalysis(), name='test')
    node.inputs.ID = '002'
    node.inputs.network = 'Default'
    node.inputs.thr = 0.95
    node.inputs.conn_model = 'cov'
    node.inputs.norm = 1
    node.inputs.prune = 1

    est_path = f"{base_dir}/miscellaneous/0021001_rsn-Default_nodetype-parc_est-sps_thrtype-DENS_thr-0.19.npy"
    est_path_tmp = f"{outdir}/0021001_rsn-Default_nodetype-parc_est-sps_thrtype-DENS_thr-0.19.npy"
    copyfile(est_path, est_path_tmp)
    node.inputs.est_path = est_path_tmp

    out = node.run()

    assert os.path.isfile(out.outputs.out_path_neat)

    temp_dir.cleanup()


def test_CombineOutputs():

    base_dir = str(Path(__file__).parent/"examples")

    node = pe.Node(interfaces.CombineOutputs(), name='test')
    node.inputs.network = None
    node.inputs.ID = '002'
    node.inputs.plot_switch = False
    node.inputs.multi_nets = None
    node.inputs.multimodal = False

    node.inputs.net_mets_csv_list = [f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.3_net_mets.csv",
                                     f"{base_dir}/miscellaneous/0021001_modality-dwi_nodetype-parc_est-csa_thrtype-PROP_thr-0.2_net_mets.csv"]

    out = node.run()
    assert out.outputs.combination_complete is True


def test_IndividualClustering():

    temp_dir = tempfile.TemporaryDirectory()
    outdir = str(temp_dir.name)

    base_dir = str(Path(__file__).parent/"examples")

    node = pe.Node(interfaces.IndividualClustering(), name='test')

    node.inputs.func_file = f"{base_dir}/BIDS/sub-25659/ses-1/func/sub-25659_ses-1_task-rest_space-MNI152NLin6Asym_desc-" \
                            f"smoothAROMAnonaggr_bold_short.nii.gz"
    node.inputs.clust_mask = f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    node.inputs.ID = '25659'
    node.inputs.k = 2
    node.inputs.clust_type = "kmeans"
    node.inputs.vox_size = "2mm"
    node.inputs.local_corr = 'allcorr'
    node.inputs.mask = f"{base_dir}/BIDS/sub-25659/ses-1/anat/sub-25659_desc-brain_mask.nii.gz"
    node.inputs.outdir = outdir

    out = node.run()

    assert isinstance(out.outputs.uatlas, str)
    assert isinstance(out.outputs.atlas, str)
    assert out.outputs.clustering is True
    assert out.outputs.clust_mask == f"{base_dir}/miscellaneous/pDMN_3_bin.nii.gz"
    assert out.outputs.k == 2
    assert out.outputs.clust_type == 'kmeans'
    assert out.outputs.func_file == node.inputs.func_file

    temp_dir.cleanup()


def test_ExtractTimeseries():

    temp_dir = tempfile.TemporaryDirectory()
    outdir = str(temp_dir.name)
    base_dir = str(Path(__file__).parent/"examples")

    func_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    img_data = np.random.rand(50, 50, 50, 20)
    img = nib.Nifti1Image(img_data, np.eye(4))
    img.to_filename(func_file.name)

    # Create a temp parcel file
    parcels_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    parcels = np.zeros((50, 50, 50))
    parcels[10:20, 0, 0], parcels[0, 10:20, 0], parcels[0, 0, 10:20] = 1, 2, 3
    nib.Nifti1Image(parcels, np.eye(4)).to_filename(parcels_tmp.name)
    net_parcels_map_nifti_file = parcels_tmp.name

    # Create empty mask file
    mask_tmp = tempfile.NamedTemporaryFile(mode='w+', suffix='.nii.gz')
    mask = np.zeros((50, 50, 50))
    nib.Nifti1Image(parcels, np.eye(4)).to_filename(mask_tmp.name)
    mask = mask_tmp.name

    # Create confound file
    conf_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv')
    conf_mat = np.random.rand(20)
    conf_df = pd.DataFrame({'Conf1': conf_mat, "Conf2": [np.nan]*len(conf_mat)})
    conf_df.to_csv(conf_file.name, sep='\t', index=False)
    conf = conf_file.name

    node = pe.Node(interfaces.ExtractTimeseries(), name='test')

    node.inputs.dir_path = outdir
    node.inputs.func_file = func_file.name
    node.inputs.mask = mask
    node.inputs.net_parcels_nii_path = net_parcels_map_nifti_file
    node.inputs.smooth = 1
    node.inputs.network = 'Default'
    node.inputs.ID = '002'
    node.inputs.smooth = 2
    node.inputs.hpass = 100
    node.inputs.coords = [[10]*3, [15]*3, [20]*3]
    node.inputs.node_size = 2
    node.inputs.extract_strategy = 'mean'

    node.inputs.roi = 'path/to/some/file'
    node.inputs.labels = [1, 2, 3]
    node.inputs.atlas = None
    node.inputs.uatlas = None

    out = node.run()

    assert np.shape(out.outputs.ts_within_nodes) == (20, 3)
    assert out.outputs.node_size == 'parc'
    assert out.outputs.smooth == 2
    assert out.outputs.atlas == None
    assert out.outputs.dir_path == outdir
    assert out.outputs.atlas == None
    assert out.outputs.uatlas == None
    assert out.outputs.labels == [1, 2, 3]
    assert out.outputs.coords == [[10, 10, 10], [15, 15, 15], [20, 20, 20]]
    assert out.outputs.hpass == 100.0
    assert out.outputs.roi == 'path/to/some/file'
    assert out.outputs.extract_strategy == 'mean'

    temp_dir.cleanup()


def test_PlotStruct():

    temp_dir = tempfile.TemporaryDirectory()
    outdir = str(temp_dir.name)

    node = pe.Node(interfaces.PlotStruct(), name='test')

    node.inputs.conn_matrix = np.ones((3, 3))
    node.inputs.conn_model = 'corr'
    node.inputs.atlas = None
    node.inputs.dir_path = outdir
    node.inputs.ID = '002'
    node.inputs.network = 'Default'
    node.inputs.labels = [1, 2, 3]
    node.inputs.roi = 'path/to/some/file'
    node.inputs.coords = [[10, 10, 10], [15, 15, 15], [20, 20, 20]]
    node.inputs.thr = 0.5
    node.inputs.node_size = 2
    node.inputs.edge_threshold = 0.5
    node.inputs.prune = True
    node.inputs.uatlas = None
    node.inputs.target_samples = 1
    node.inputs.norm = False
    node.inputs.binary = False
    node.inputs.track_type = None
    node.inputs.directget = None
    node.inputs.min_length = 1

    out = node.run()

    assert out.outputs.out == 'None'
    temp_dir.close()


def test_PlotFunc():

    temp_dir = tempfile.TemporaryDirectory()
    outdir = str(temp_dir.name)

    node = pe.Node(interfaces.PlotFunc(), name='test')

    node.inputs.conn_matrix = np.ones((3, 3))
    node.inputs.conn_model = 'corr'
    node.inputs.atlas = None
    node.inputs.dir_path = outdir
    node.inputs.ID = '002'
    node.inputs.network = 'Default'
    node.inputs.labels = [1, 2, 3]
    node.inputs.roi = 'path/to/some/file'
    node.inputs.coords = [[10, 10, 10], [15, 15, 15], [20, 20, 20]]
    node.inputs.thr = 0.5
    node.inputs.node_size = 2
    node.inputs.edge_threshold = 0.5
    node.inputs.smooth = 2
    node.inputs.prune = True
    node.inputs.uatlas = None
    node.inputs.norm = False

    node.inputs.binary = False
    node.inputs.hpass = 100
    node.inputs.extract_strategy = 'strategy'
    node.inputs.edge_color_override = False

    out = node.run()

    assert out.outputs.out == 'None'


"""
def test_RegisterDWI():

    fa_path = File(exists=True, mandatory=True)
    ap_path = File(exists=True, mandatory=True)
    B0_mask = File(exists=True, mandatory=True)
    anat_file = File(exists=True, mandatory=True)
    gtab_file = File(exists=True, mandatory=True)
    dwi_file = File(exists=True, mandatory=True)
    in_dir = traits.Any()
    vox_size = traits.Str("2mm", mandatory=True, usedefault=True)
    template_name = traits.Str("MNI152_T1", mandatory=True, usedefault=True)
    mask = traits.Any(mandatory=False)
    simple = traits.Bool(False, usedefault=True)
    overwrite = traits.Bool(False, usedefault=True)


    temp_dir = tempfile.TemporaryDirectory()
    base_dir = str(Path(__file__).parent/"examples")
    outdir = str(temp_dir.name)

    # No testing data for these inputs
    fa_path = base_dir + '/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz'
    ap_path = base_dir + '/BIDS/sub-25659/ses-1/dwi/final_preprocessed_dwi.nii.gz'

    B0_mask = base_dir + '/003/dmri/sub-003_b0_brain_mask.nii.gz'
    anat_file = base_dir + '/003/anat/sub-003_T1w.nii.gz'
    gtab_file =
    node = pe.Node(interfaces.RegisterDWI(), name='test')
"""
