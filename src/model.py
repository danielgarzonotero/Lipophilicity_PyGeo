import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, NNConv
from torch_geometric.nn import aggr

class GCN_Geo(torch.nn.Module):
    def __init__(self, 
                 initial_dim_gcn, 
                 edge_dim_feature,
                 hidden_dim_nn_1=100,
                 hidden_dim_nn_2=20,
                 hidden_dim_fcn_1=50):
        super(GCN_Geo, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)), 
                                aggr='add' )
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)), 
                                aggr='add')

        self.readout = aggr.SumAggregation()

        self.linear1 = nn.Linear(hidden_dim_nn_2, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.nn_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.nn_conv_2(x, edge_index, edge_attr)
        x = F.relu(x)

        x = self.readout(x, data.batch)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)

        return x.view(-1,)
    
    


''' class GCN_Geo(torch.nn.Module):
    def __init__(self, initial_dim_gcn=25, 
                hidden_dim_gat_1 = 500,
                hidden_dim_gat_2 = 300,
                 hidden_dim_fcn_1 = 100,
                 ):
        super(GCN_Geo, self).__init__()
        
        self.gat_conv_1 = GATConv(initial_dim_gcn,  hidden_dim_gat_1)
        self.gat_conv_2 = GATConv(hidden_dim_gat_1,  hidden_dim_gat_2)
        
        self.readout = aggr.SumAggregation()
        
        self.linear1 = nn.Linear(hidden_dim_gat_2, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, 1)

    def forward(self, data):
        # Message passing layers:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.gat_conv_1(x, edge_index, edge_attr)  
        x = F.relu(x)
        x = self.gat_conv_2(x, edge_index, edge_attr)  
        x = F.relu(x)
        
        x = self.readout(x, data.batch)
        
        
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x.view(-1,)
 '''


""" import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import aggr


class GCN_Geo(torch.nn.Module):
    def __init__(self, initial_dim_gcn, hidden_dim_gcn, hidden_dim_fcn):
        super(GCN_Geo, self).__init__()
        self.conv1 = GCNConv(initial_dim_gcn, hidden_dim_gcn)
        self.conv2 = GCNConv(hidden_dim_gcn, hidden_dim_gcn)
        
        self.readout = aggr.SumAggregation()
        
        self.linear1 = nn.Linear(hidden_dim_gcn, hidden_dim_fcn)
        self.linear2 = nn.Linear(hidden_dim_fcn, 1)

    def forward(self, data):
        # Message passing layers:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Aggregate to get molecule level features:
        #En este caso, se utiliza la clase aggr.SumAggregation() para realizar 
        #la agregación mediante la suma de las características de los nodos. 
        #Esto significa que para cada molécula en el grafo, se suman las 
        #características de todos los nodos que pertenecen a esa molécula. 
        #El resultado de esta suma es una representación agregada de la molécula.
        #La capa de agregación self.readout toma dos argumentos: x y data.batch. 
        #El argumento x es el tensor de características de los nodos después de 
        #pasar por las capas GCN. El argumento data.batch es un tensor que asigna 
        #a cada nodo del grafo su correspondiente molécula. Esto se hace mediante 
        #el uso de un atributo especial llamado batch en el objeto data que se pasa 
        #como argumento al método forward().
        #La capa de agregación self.readout utiliza el tensor x y el 
        #tensor data.batch para realizar la agregación. Para cada molécula,
        #se suman las características de los nodos correspondientes y se obtiene una 
        #única representación a nivel de molécula. 
        
        x = self.readout(x, data.batch)
        
        # FCNN to predict molecular property:
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        
        return x.view(-1,) 


 """