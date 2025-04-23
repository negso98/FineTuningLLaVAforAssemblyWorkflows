import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

dot = Digraph()
dot.node('A', 'Data Collection')
dot.node('B', 'Model Training')
dot.node('C', 'Evaluation & Testing')
dot.node('D', 'Deployment & Integration')



dot.edges(['AB', 'BC', 'CD'])

dot.render('markov_model', format='png', view=True)

G = nx.DiGraph()
G.add_edges_from([
    ('Data Collection', 'Model Training'),
    ('Model Training', 'Evaluation & Testing'),
    ('Evaluation & Testing', 'Deployment & Integration')
])

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color='gray', font_size=8)
plt.show()
