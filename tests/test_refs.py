﻿#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:19:14 2017
@authors: Derek Pisner & Ryan Hammonds
"""
import nibabel as nib
from pathlib import Path
import pkg_resources
import logging

logger = logging.getLogger(__name__)
logger.setLevel(50)


def test_bigsmallref():
    """
    Test big and small ref files for existence and non-corruption
    """
    bigref1mm = nib.load(pkg_resources.resource_filename("pynets", f"rsnrefs/BIGREF1mm.nii.gz"))
    bigref2mm = nib.load(pkg_resources.resource_filename("pynets", f"rsnrefs/BIGREF2mm.nii.gz"))
    smallref1mm = nib.load(pkg_resources.resource_filename("pynets", f"rsnrefs/SMALLREF1mm.nii.gz"))
    smallref2mm = nib.load(pkg_resources.resource_filename("pynets", f"rsnrefs/SMALLREF2mm.nii.gz"))
    assert bigref1mm is not None
    assert bigref2mm is not None
    assert smallref1mm is not None
    assert smallref2mm is not None


def test_atlases():
    """
    Test atlas files for existence and non-corruption
    """
    whole_brain_cluster_labels_PCA200 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/whole_brain_cluster_labels_PCA200.nii.gz"))
    AAL2zourioMazoyer2002 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/AAL2zourioMazoyer2002.nii.gz"))
    AALTzourioMazoyer2002 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/AALTzourioMazoyer2002.nii.gz"))
    AICHAJoliot2015 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/AICHAJoliot2015.nii.gz"))
    AICHAreorderedJoliot2015 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/AICHAreorderedJoliot2015.nii.gz"))
    BrainnetomeAtlasFan2016 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/BrainnetomeAtlasFan2016.nii.gz"))
    CorticalAreaParcellationfromRestingStateCorrelationsGordon2014 = nib.load(
        pkg_resources.resource_filename("pynets", f"core/atlases/CorticalAreaParcellationfromRestingStateCorrelationsGordon2014.nii.gz"))
    DesikanKlein2012 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/DesikanKlein2012.nii.gz"))
    destrieux2009_rois = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/destrieux2009_rois.nii.gz"))
    Hammers_mithAtlasn30r83Hammers2003Gousias2008 = nib.load(
        pkg_resources.resource_filename("pynets", f"core/atlases/Hammers_mithAtlasn30r83Hammers2003Gousias2008.nii.gz"))
    HarvardOxfordThr252mmWholeBrainMakris2006 = nib.load(
        pkg_resources.resource_filename("pynets", f"core/atlases/HarvardOxfordThr252mmWholeBrainMakris2006.nii.gz"))
    Juelichgmthr252mmEickhoff2005 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/Juelichgmthr252mmEickhoff2005.nii.gz"))
    MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics = nib.load(
        pkg_resources.resource_filename("pynets", f"core/atlases/MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics.nii.gz"))
    RandomParcellationsc05meanalll43Craddock2011 = nib.load(
        pkg_resources.resource_filename("pynets", f"core/atlases/RandomParcellationsc05meanalll43Craddock2011.nii.gz"))
    VoxelwiseParcellationt058kLeadDBS = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/VoxelwiseParcellationt058kLeadDBS.nii.gz"))
    VoxelwiseParcellationt0435kLeadDBS = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/VoxelwiseParcellationt0435kLeadDBS.nii.gz"))
    VoxelwiseParcellationt0515kLeadDBS = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/VoxelwiseParcellationt0515kLeadDBS.nii.gz"))
    whole_brain_cluster_labels_PCA100 = nib.load(pkg_resources.resource_filename("pynets", f"core/atlases/whole_brain_cluster_labels_PCA100.nii.gz"))
    assert whole_brain_cluster_labels_PCA200 is not None
    assert AAL2zourioMazoyer2002 is not None
    assert AALTzourioMazoyer2002 is not None
    assert AICHAJoliot2015 is not None
    assert AICHAreorderedJoliot2015 is not None
    assert BrainnetomeAtlasFan2016 is not None
    assert CorticalAreaParcellationfromRestingStateCorrelationsGordon2014 is not None
    assert DesikanKlein2012 is not None
    assert destrieux2009_rois is not None
    assert Hammers_mithAtlasn30r83Hammers2003Gousias2008 is not None
    assert HarvardOxfordThr252mmWholeBrainMakris2006 is not None
    assert Juelichgmthr252mmEickhoff2005 is not None
    assert MICCAI2012MultiAtlasLabelingWorkshopandChallengeNeuromorphometrics is not None
    assert RandomParcellationsc05meanalll43Craddock2011 is not None
    assert VoxelwiseParcellationt058kLeadDBS is not None
    assert VoxelwiseParcellationt0435kLeadDBS is not None
    assert VoxelwiseParcellationt0515kLeadDBS is not None
    assert whole_brain_cluster_labels_PCA100 is not None


def test_templates():
    """
    Test template files for existence and non-corruption
    """
    ch2better = nib.load(pkg_resources.resource_filename("pynets", f"templates/ch2better.nii.gz"))

    FA_2mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/FA_2mm.nii.gz"))
    MNI152_T1_brain_2mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/MNI152_T1_brain_2mm.nii.gz"))
    CorpusCallosum_2mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/CorpusCallosum_2mm.nii.gz"))
    LateralVentricles_2mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/LateralVentricles_2mm.nii.gz"))
    MNI152_T1_brain_mask_2mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/MNI152_T1_brain_mask_2mm.nii.gz"))

    FA_1mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/FA_1mm.nii.gz"))
    MNI152_T1_brain_1mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/MNI152_T1_brain_1mm.nii.gz"))
    CorpusCallosum_1mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/CorpusCallosum_1mm.nii.gz"))
    LateralVentricles_1mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/LateralVentricles_1mm.nii.gz"))
    MNI152_T1_brain_mask_1mm = nib.load(pkg_resources.resource_filename("pynets", f"templates/MNI152_T1_brain_mask_1mm.nii.gz"))

    assert ch2better is not None
    assert FA_2mm is not None
    assert MNI152_T1_brain_2mm is not None
    assert CorpusCallosum_2mm is not None
    assert LateralVentricles_2mm is not None
    assert MNI152_T1_brain_mask_2mm is not None

    assert FA_1mm is not None
    assert MNI152_T1_brain_1mm is not None
    assert CorpusCallosum_1mm is not None
    assert LateralVentricles_1mm is not None
    assert MNI152_T1_brain_mask_1mm is not None
