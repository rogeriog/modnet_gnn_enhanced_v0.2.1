""" This submodule contains the GNN featurizers implementation
as described in the paper `Boosting feature-based machine learning
 models for materials science:` """

import os
import numpy as np
import pandas as pd
import pickle
import modnet.featurizers
import contextlib
from urllib.request import urlretrieve
from sklearn.model_selection import KFold
from megnet.data.crystal import CrystalGraph
from megnet.models import MEGNetModel
from keras.models import load_model
import json
from keras.callbacks import EarlyStopping
from keras.models import Model

### FUNCTIONS TO GET MEGNet FEATURES
from ..GNN_encoders.megnet_functions import get_MEGNetFeatures, get_MEGNetBaseFeatures, get_MEGNetAdjacentFeatures

class MEGNetFeaturizer:
    def __init__(self, model_type, **kwargs):
        self.model_type = model_type
        self.parent_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        self.parent_dir = os.path.abspath(self.parent_dir)
        self.file_path = os.path.join(self.parent_dir, f'MEGNetModel__{self.model_type}.h5')        
        self.adjacent_model_path = kwargs.get('adjacent_model_path', '.')
        if self.model_type == 'MatMinerEncoded_v1':
            # check if model file is downloaded
            if os.path.isfile(self.file_path):
                print('MEGNetModel__MatMinerEncoded_v1.h5 found in parent directory')
            else:  # download from url not implemented
                download_url = 'https://figshare.com/ndownloader/files/36767340'
                urlretrieve(download_url, self.file_path)
                print('MEGNetModel__MatMinerEncoded_v1.h5 downloaded from figshare')
        elif self.model_type == 'OFMEncoded_v1':
            # check if model file is downloaded
            if os.path.isfile(self.file_path):
                print('MEGNetModel__OFMEncoded_v1.h5 found in parent directory')
            else: # not implemented
                download_url = 'https://figshare.com/ndownloader/files/36767340'
                urlretrieve(download_url, self.file_path)
                print('MEGNetModel__OFMEncoded_v1.h5 downloaded from figshare')
        
    def get_features(self, structures, targets=None):
        if self.model_type == 'MatMinerEncoded_v1':
            file_path_without_ext, _ = os.path.splitext(self.file_path)
            self.scaler_path = file_path_without_ext + "_scaler.pkl"
            n_targets=758 ## have to get from the resulting data, is fixed value.
            df = get_MEGNetFeatures(structures, model_name=self.model_type, n_targets=n_targets,
                    neuron_layers=(64,128,64), model_file=self.file_path,
                    scaler_file=self.scaler_path)
            return df
        elif self.model_type == 'OFMEncoded_v1':
            file_path_without_ext, _ = os.path.splitext(self.file_path)
            self.scaler_path = file_path_without_ext + "_scaler.pkl"
            n_targets=188
            df = get_MEGNetFeatures(structures, model_name=self.model_type, n_targets=n_targets,
                    neuron_layers=(64,128,64), model_file=self.file_path,
                    scaler_file=self.scaler_path)
            return df
        elif self.model_type == 'pretrained_models':
            df = get_MEGNetBaseFeatures(structures)
            return df
        elif self.model_type == 'adjacent':
            # check if model file is present
            # remove from self.file_path the path, to check on execution folder.
            # self.file_path = os.path.basename(self.file_path)
            if os.path.isfile(self.adjacent_model_path):
                print('MEGNetModel__adjacent.h5 found.')
            else:
                raise ValueError('MEGNetModel__adjacent.h5 not found.')
            df = get_MEGNetAdjacentFeatures(structures, model_file=self.adjacent_model_path)
            return df
        else:
            raise ValueError('model_type not recognized')
    


class MODNetOmega2024(modnet.featurizers.MODFeaturizer):
    """
    ModnetOMEGA - _O_rbitalFieldMatrix + _MEG_Net pretrained models + _A_djacent MEGNet model
    This featurizer enhances MODNet capabilities to include the more complex and
    computationally intensive OrbitalFieldMatrix in a encoded manner, MEGNet pretrained
    models included as features taking an intermediate layer of the model, and
    an adjacent MEGNet model trained on-the-fly on the target to capture other patterns.
    In detail this featurizer includes:
    1. OrbitalFieldMatrix features derived from the mp-gap dataset, encoded using a regular
    autoencoder, and predicted by a MEGNet model.
    2. Pretrained MEGNet models for various properties, such as band gap, formation energy,
    elastic properties, etc., provided by the MEGNet team.
    3. Optionally, users can train an adjacent MEGNet model on-the-fly to predict a new set
    of features, ensuring greater flexibility and mitigating potential bias from the
    materials project data, which may not be representative of the target dataset.
    
    NOTE: The featurizer currently supports only structure inputs.
    """

    def __init__(self, adjacent_model: bool = False, targets=None, structures=None, **kwargs):
        """Creates the featurizer and imports all featurizer functions.

        Parameters:
            adjacent_model: Whether to train an adjacent MEGNet model on-the-fly to predict
            a new set of features. This is useful for ensuring greater flexibility and mitigating
            potential bias from the materials project data, which may not be representative of the
            target dataset.
            targets: A list of target values for the training set. Only required if `adjacent_model`
            is True.
            structures: A list of structures for the training set. Only required if `adjacent_model`
            is True. 

            kwargs: Additional keyword arguments to pass to the MEGNet model.
            adjacent_model_path: Path to the adjacent model. If not provided, the model will be
            will be read from the current working directory.
        """

        super().__init__()
        self.adjacent_model = adjacent_model
        self.adjacent_model_path = kwargs.get("adjacent_model_path", '.')
        self.adjacent_model_path = os.path.join(self.adjacent_model_path, 'MEGNetModel__adjacent.h5')
        self.load_featurizers()
        if self.adjacent_model:
            if not os.path.isfile(self.adjacent_model_path):
                self.train_MEGNet_on_the_fly(structures, targets, **kwargs)

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):
            
            from pymatgen.analysis.local_env import VoronoiNN
            from matminer.featurizers.composition import (
                AtomicOrbitals,
                AtomicPackingEfficiency,
                BandCenter,
                # CohesiveEnergy, - This descriptor was not used in the paper preset
                # ElectronAffinity, - This descriptor was not used in the paper preset
                ElectronegativityDiff,
                ElementFraction,
                ElementProperty,
                IonProperty,
                Miedema,
                OxidationStates,
                Stoichiometry,
                TMetalFraction,
                ValenceOrbital,
                YangSolidSolution,
            )
            from matminer.featurizers.structure import (
                # BagofBonds, - This descriptor was not used in the paper preset
                BondFractions,
                ChemicalOrdering,
                CoulombMatrix,
                DensityFeatures,
                EwaldEnergy,
                GlobalSymmetryFeatures,
                MaximumPackingEfficiency,
                # PartialRadialDistributionFunction,
                RadialDistributionFunction,
                SineCoulombMatrix,
                StructuralHeterogeneity,
                XRDPowderPattern,
            )

            from matminer.featurizers.site import (
                AGNIFingerprints,
                AverageBondAngle,
                AverageBondLength,
                BondOrientationalParameter,
                ChemEnvSiteFingerprint,
                CoordinationNumber,
                CrystalNNFingerprint,
                GaussianSymmFunc,
                GeneralizedRadialDistributionFunction,
                LocalPropertyDifference,
                OPSiteFingerprint,
                VoronoiFingerprint,
            )

            self.composition_featurizers = (
                AtomicOrbitals(),
                AtomicPackingEfficiency(),
                BandCenter(),
                ElementFraction(),
                ElementProperty.from_preset("magpie"),
                IonProperty(),
                Miedema(),
                Stoichiometry(),
                TMetalFraction(),
                ValenceOrbital(),
                YangSolidSolution(),
            )

            self.oxid_composition_featurizers = (
                ElectronegativityDiff(),
                OxidationStates(),
            )

            ## OFM encoded model and pretrained MEGNet models
            MEGNetOFMFeaturizer = MEGNetFeaturizer(model_type='OFMEncoded_v1')
            MEGNetPretrainedFeaturizer = MEGNetFeaturizer(model_type='pretrained_models')
            self.megnet_featurizers = [
                MEGNetOFMFeaturizer,
                MEGNetPretrainedFeaturizer,
            ]

            self.structure_featurizers = [
                DensityFeatures(),
                GlobalSymmetryFeatures(),
                RadialDistributionFunction(),
                CoulombMatrix(),
                # PartialRadialDistributionFunction(),
                SineCoulombMatrix(),
                EwaldEnergy(),
                BondFractions(),
                StructuralHeterogeneity(),
                MaximumPackingEfficiency(),
                ChemicalOrdering(),
                XRDPowderPattern(),
                # BagofBonds(),
            ]

            # Patch for matminer: see https://github.com/hackingmaterials/matminer/issues/864
            self.structure_featurizers[0].desired_features = None
            self.structure_featurizers[1].desired_features = None

            self.site_featurizers = (
                AGNIFingerprints(),
                AverageBondAngle(VoronoiNN()),
                AverageBondLength(VoronoiNN()),
                BondOrientationalParameter(),
                ChemEnvSiteFingerprint.from_preset("simple"),
                CoordinationNumber(),
                CrystalNNFingerprint.from_preset("ops"),
                GaussianSymmFunc(),
                GeneralizedRadialDistributionFunction.from_preset("gaussian"),
                LocalPropertyDifference(),
                OPSiteFingerprint(),
                VoronoiFingerprint(),
            )

            if self.adjacent_model:
                MEGNetAdjacent = MEGNetFeaturizer(model_type='adjacent', adjacent_model_path=self.adjacent_model_path)
                self.megnet_featurizers.append(MEGNetAdjacent)
            

    def featurize_composition(self, df):
        """Applies the preset composition featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        from pymatgen.core.periodic_table import Element

        df = super().featurize_composition(df)

        _orbitals = {"s": 1, "p": 2, "d": 3, "f": 4}
        df["AtomicOrbitals|HOMO_character"] = df["AtomicOrbitals|HOMO_character"].map(
            _orbitals
        )
        df["AtomicOrbitals|LUMO_character"] = df["AtomicOrbitals|LUMO_character"].map(
            _orbitals
        )

        df["AtomicOrbitals|HOMO_element"] = df["AtomicOrbitals|HOMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )
        df["AtomicOrbitals|LUMO_element"] = df["AtomicOrbitals|LUMO_element"].apply(
            lambda x: -1 if not isinstance(x, str) else Element(x).Z
        )

        return modnet.featurizers.clean_df(df)

    def featurize_structure(self, df):
        """Applies the preset structural featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """
        # featurizes matminer features
        if self.structure_featurizers:
            df_mm = super().featurize_structure(df.copy())
        
        # featurizes with the megnet models
        dfs = [df_mm]
        for featurizer in self.megnet_featurizers:
            df_tmp = featurizer.get_features(df['structure'])
            dfs.append(df_tmp)
        df = pd.concat(dfs, axis=1)

        _crystal_system = {
            "cubic": 1,
            "tetragonal": 2,
            "orthorombic": 3,
            "hexagonal": 4,
            "trigonal": 5,
            "monoclinic": 6,
            "triclinic": 7,
        }

        def _int_map(x):
            if x == np.nan:
                return 0
            elif x:
                return 1
            else:
                return 0

        df["GlobalSymmetryFeatures|crystal_system"] = df[
            "GlobalSymmetryFeatures|crystal_system"
        ].map(_crystal_system)
        df["GlobalSymmetryFeatures|is_centrosymmetric"] = df[
            "GlobalSymmetryFeatures|is_centrosymmetric"
        ].map(_int_map)

        return modnet.featurizers.clean_df(df)

    def featurize_site(self, df):
        """Applies the preset site featurizers to the input dataframe,
        renames some fields and cleans the output dataframe.

        """

        # rename some features for backwards compatibility with pretrained models
        aliases = {
            "GeneralizedRadialDistributionFunction": "GeneralizedRDF",
            "AGNIFingerprints": "AGNIFingerPrint",
            "BondOrientationalParameter": "BondOrientationParameter",
        }
        df = super().featurize_site(df, aliases=aliases)
        df = df.loc[:, (df != 0).any(axis=0)]

        return modnet.featurizers.clean_df(df)


    def train_MEGNet_on_the_fly(self, structures, targets, **kwargs):
        # apply a scaler to the targets
        from sklearn.preprocessing import MinMaxScaler
        targets = np.array(targets)
        targets = targets.reshape(-1,1)
        scaler = MinMaxScaler()
        targets = scaler.fit_transform(targets)
        # save scaler to pickle
        pickle.dump(scaler, open('MEGNetModel__adjacent_scaler.pkl', 'wb'))
        print('Scaler of the targets for adjacent model saved to MEGNetModel__adjacent_scaler.pkl')
        # train a MEGNet model on the fly to predict a new set of features
        max_epochs=kwargs.get('max_epochs',100)
        patience = kwargs.get('patience',10)
        n1=kwargs.get('n1', 64) 
        n2=kwargs.get('n2', 32) 
        n3=kwargs.get('n3', 16)
        nfeat_bond = kwargs.get('nfeat_bond', 100)
        r_cutoff = kwargs.get('r_cutoff', 5)
        gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
        gaussian_width = kwargs.get('gaussian_width', 0.5)
        graph_converter = CrystalGraph(cutoff=r_cutoff)
        early_stopping = EarlyStopping(monitor='val_mae',patience=patience, restore_best_weights=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        # we use just one k fold to get the validation set as criteria for convergence
        train_index, val_index = list(kf.split(structures))[0]
        train_structures, val_structures = [structures.iloc[i] for i in train_index], [structures.iloc[i] for i in val_index]
        train_targets, val_targets = [targets[i] for i in train_index], [targets[i] for i in val_index]

        model = MEGNetModel(metrics=['mae'], graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                            )
        model.train(train_structures, train_targets, validation_structures=val_structures, 
                    validation_targets=val_targets, epochs=max_epochs, save_checkpoint=False, callbacks=[early_stopping])
        model.save(f'MEGNetModel__adjacent.h5')
        print('MEGNet model on the fly saved to MEGNetModel__adjacent.h5')
        
class MODNetOmegaFast2024(MODNetOmega2024):
    """This featurizer aims to rapidly screen large datasets by bypassing
    the computationally intensive step of applying matminer features directly.
    Currently, it only supports structure-based inputs. Besides the features
    already described in `MODNetOmega2024`, this featurizer also includes:
    > Compressed MatMiner features extracted from the PPdebreuck2020Featurizer and applied
    to the mp-gap dataset. These features are predicted using a MEGNet model trained on
    the encoded features of that dataset.
    """   

    def load_featurizers(self):
        with contextlib.redirect_stdout(None):

            MEGNetMMFeaturizer = MEGNetFeaturizer(model_type='MatMinerEncoded_v1')
            MEGNetOFMFeaturizer = MEGNetFeaturizer(model_type='OFMEncoded_v1')
            MEGNetPretrainedFeaturizer = MEGNetFeaturizer(model_type='pretrained_models')

            self.megnet_featurizers = [
                MEGNetMMFeaturizer,
                MEGNetOFMFeaturizer,
                MEGNetPretrainedFeaturizer,
            ]
            if self.adjacent_model:
                MEGNetAdjacent = MEGNetFeaturizer(model_type='adjacent')
                # append 
                self.megnet_featurizers.append(MEGNetAdjacent)
            
            # to trigger featurize_structure
            self.structure_featurizers = tuple(self.megnet_featurizers)

    def featurize_structure(self, df):
        dfs = []
        for featurizer in self.megnet_featurizers:
            df_tmp = featurizer.get_features(df['structure'])
            dfs.append(df_tmp)
        df = pd.concat(dfs, axis=1)
        return modnet.featurizers.clean_df(df)      




        
    