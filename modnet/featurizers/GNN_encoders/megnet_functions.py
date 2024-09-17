import numpy as np
from keras.models import Model
import warnings
from pickle import load
import pandas as pd
import os
from typing import Tuple, Any

warnings.filterwarnings("ignore")
from megnet.utils.models import load_model, AVAILABLE_MODELS
from megnet.models import MEGNetModel
from megnet.data.crystal import CrystalGraph
# print(AVAILABLE_MODELS)

### FUNCTIONS TO SETUP AND EVALUATE MEGNET MODELS


def model_setup(ntarget: int = None,
                **kwargs) -> Any:
    """
    This function takes in a number of optional parameters for creating a MEGNet model, such as number of neurons 
    in different layers, and the number of features for bonds.
    It returns an instance of a MEGNet model which is set up with the given parameters.
    """
    ## default architecture:
    n1=kwargs.get('n1', 64) 
    n2=kwargs.get('n2', 32) 
    n3=kwargs.get('n3', 16)
    nfeat_bond = kwargs.get('nfeat_bond', 100)
    r_cutoff = kwargs.get('r_cutoff', 5)
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = kwargs.get('gaussian_width', 0.5)
    graph_converter = CrystalGraph(cutoff=r_cutoff)

    model = MEGNetModel(graph_converter=graph_converter, centers=gaussian_centers, width=gaussian_width,
                        ntarget=ntarget, **kwargs)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)
    print(short_model_summary.splitlines()[-4])
    return model

def load_model_scaler(id: str = '',
                      n_targets: int = 1 ,
                      neuron_layers: Tuple[int] = (64,32,16),
                      **kwargs) -> Tuple[Any, Any]:
    """
    This function takes in an id, number of targets, a mode, and other optional parameters for loading a previously trained MEGNet model and its corresponding scaler.
    It returns a tuple of the loaded model and scaler.
    """
    n1,n2,n3=neuron_layers
    model = model_setup(ntarget=n_targets, n1=n1, n2=n2, n3=n3,
                        **kwargs)
    modelpath_id = kwargs.get("modeldir", "./")+id
    model_file=kwargs.get('model_file',f"{modelpath_id}_weights.h5")
    scaler_file=kwargs.get('scaler_file',f'{modelpath_id}_scaler.pkl')
    model.load_weights(model_file)
    try: ## if scaler not found, it will be None
        scaler = load(open(scaler_file, 'rb'))
    except:
        scaler = None
    return (model, scaler)

def megnet_evaluate_structures(model, structures,
                               targets=None,
                               scaler=None, **kwargs):

    labels = kwargs.get('labels', ['']*len(structures))

    noTargets=False
    if targets is None:
        target_values = np.array([1]*len(structures))
        noTargets=True
    else:
        if isinstance(targets, pd.DataFrame):
            target_values=targets.values
        else:
            target_values=targets
    # have to exclude structures that dont form compatible graphs and their corresponding targets.
    structures_valid = []
    targets_valid = []
    labels_valid = []
    structures_invalid = []
    for s, p, l in zip(structures, target_values, labels):
        try:
            graph = model.graph_converter.convert(s)
            structures_valid.append(s)
            if scaler is not None:
                targets_valid.append(np.nan_to_num(
                    scaler.transform(p.reshape(1, -1))))
            else:
                targets_valid.append(p)
            labels_valid.append(l)
        except:
            structures_invalid.append(s)
    # structures_valid = np.array(structures_valid)

    y = np.array(targets_valid)
    y = y.squeeze()
    labels = np.array(labels_valid)
    print(f"Following invalid structures: {structures_invalid}.")
    # print(type(structures_valid),structures_valid)
    ypred = model.predict_structures(list(structures_valid))
    if noTargets:
        return (structures_valid,ypred)
    if not noTargets:
        return (structures_valid,ypred, y, labels)
    # y_pred=y_pred.flatten()

### FUNCTIONS TO OBTAIN FEATURE DATAFRAMES FROM MEGNET MODELS
def get_MEGNetBaseFeatures(structures):
    MEGNetBaseFeats=[] # 'Eform_MP_2019',
    for model_name in ['Efermi_MP_2019','Bandgap_MP_2018','logK_MP_2019','logG_MP_2019']:
        model=load_model(model_name) 
        intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.layers[-3].output)   
        MEGNetModel_structs=[]
        indexes = structures.index.to_list()
        for original_index, s in zip(indexes, structures):
            try:
                graph = model.graph_converter.convert(s)
                inp = model.graph_converter.graph_to_input(graph)
                pred = intermediate_layer_model.predict(inp, verbose=False)
                s_model_results=pd.DataFrame([pred[0][0]], 
                                          columns=[f"{model_name}_{idx+1}" for idx in 
                                                   range(len(pred[0][0]))],
                                          index=[original_index])
                MEGNetModel_structs.append(s_model_results)
            except Exception as e:
                print(e)
                print("Probably an invalid structure was passed to the model, continuing..")
                # make a panda series of nan with size 32
                nans = [[np.nan]*32]
                columns= [f"{model_name}_{idx+1}" for idx in range(len(pred[0][0]))]
                s_model_results=pd.DataFrame(nans,columns=columns,index=[original_index])
                MEGNetModel_structs.append(s_model_results)
                continue
        ## now append the columns with the layer of each model
        MEGNetModel_structs=pd.concat(MEGNetModel_structs,axis=0)
        MEGNetBaseFeats.append(MEGNetModel_structs)
        print(f"Features calculated for model {model_name}.")
    ## now every structure calculated with each model is combined in a final dataframe
    MEGNetBaseFeats=pd.concat(MEGNetBaseFeats,axis=1)
    return MEGNetBaseFeats

# implement special case here to get on the fly features...
def get_MEGNetFeatures(structures,
                       n_targets : int = 1,
                       neuron_layers : Tuple[int] = (64,32,16), 
                       model=None, 
                       model_file=None, 
                       scaler=None,
                       scaler_file=None,
                       **kwargs):
    '''From a specified model, either passed directly or loaded from file
    scaler is optional to scale back the produced output. 
    Reads a set of structures filters them and returns a dataframe with the
    features of the model.
    '''
    model_name=kwargs.get('model_name','myMEGNetModel')
    if model is None:
        model,scaler=load_model_scaler(n_targets=n_targets, 
                        neuron_layers=neuron_layers,
                        model_file=model_file, scaler_file=scaler_file, 
                        **kwargs)

    MEGNetFeatsDF=[] 
    # get indexes from structures
    indexes = structures.index.to_list()
    # get the features from the model for each valid structure   
    structures_valid,ypred=megnet_evaluate_structures(model,structures)
    # print(ypred)
    for original_index, s in zip(indexes, structures):
        if s in list(structures_valid):
            s_idx = list(structures_valid).index(s)
            p = ypred[s_idx]
            if scaler is None:
                feat_data=pd.DataFrame([p],columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)],
                                        index=[original_index])
                struct=pd.DataFrame({'structure': [s]}, index=[original_index])
                modeldata_struct = pd.concat([struct,feat_data], axis=1)
            else:
                feat_data=pd.DataFrame(scaler.inverse_transform(p.reshape(1, -1)),
                                    columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)],
                                    index=[original_index])
                struct=pd.DataFrame({'structure': [s]}, index=[original_index])
                modeldata_struct = pd.concat([struct,feat_data], axis=1)
        else:
            feat_data=pd.DataFrame([[np.nan] * n_targets],
                                    columns=[f"MEGNet_{model_name}_{idx+1}" for idx in range(n_targets)],
                                    index=[original_index])
            struct=pd.DataFrame({'structure': [s]}, index=[original_index])
            modeldata_struct = pd.concat([struct,feat_data], axis=1)        
        MEGNetFeatsDF.append(modeldata_struct)
    MEGNetFeatsDF = pd.concat(MEGNetFeatsDF,axis=0)  
    return MEGNetFeatsDF
    
def get_MEGNetAdjacentFeatures(structures,
                       n_targets : int = 1,
                       neuron_layers : Tuple[int] = (64,32,16), 
                       model=None, 
                       model_file='MEGNetModel__adjacent.h5', 
                       scaler=None,
                       scaler_file='MEGNetModel__adjacent_scaler.pkl',
                       **kwargs):
    # get base folder if any from model_file
    base_folder=os.path.dirname(model_file)
    # set it on scaler_file
    scaler_file=os.path.join(base_folder,scaler_file)

    model_name=kwargs.pop('model_name','Adjacent')
    model,scaler=load_model_scaler(n_targets=n_targets, 
                        model_file=model_file, scaler_file=scaler_file, 
                        **kwargs)
    intermediate_layer_model = Model(inputs=model.input,
                            outputs=model.layers[-3].output)  
    
    MEGNetAdjacent_structs=[]   
    indexes = structures.index.to_list()
    for original_index, s in zip(indexes, structures):
        try:
            graph = model.graph_converter.convert(s)
            inp = model.graph_converter.graph_to_input(graph)
            pred = intermediate_layer_model.predict(inp, verbose=False)
            s_model_results=pd.DataFrame([pred[0][0]], 
                                        columns=[f"{model_name}_{idx+1}" for idx in 
                                                range(len(pred[0][0]))],
                                        index=[original_index])
            MEGNetAdjacent_structs.append(s_model_results)
        except Exception as e:
            print(e)
            print("Probably an invalid structure was passed to the model, continuing..")
            # make a panda series of nan with size 32
            nans = [[np.nan]*32]
            columns= [f"{model_name}_{idx+1}" for idx in range(len(pred[0][0]))]
            model_struct=pd.DataFrame(nans,columns=columns, index=[original_index])
            MEGNetAdjacent_structs.append(s_model_results)
            continue
    ## now append the columns with the layer of each model
    MEGNetAdjacentFeatsDF = pd.concat(MEGNetAdjacent_structs,axis=0)
    print(f"Features calculated for model {model_name}.")
    return MEGNetAdjacentFeatsDF


__all__ = ['model_setup', 'load_model_scaler', 'megnet_evaluate_structures',
           'get_MEGNetFeatures', 'get_MEGNetBaseFeatures', 'get_MEGNetAdjacentFeatures',
           ]