<H3> Name :DEEPIKA R</H3>
<H3>Register No.: 212223230038</H3>
<H3> Experiment 1</H3>
<H3>DATE:25/04/2026</H3>
<H1 ALIGN=CENTER> Implementation of Bayesian Networks</H1>
<H3> Aim :</H3>
    To create a bayesian Network for the given dataset in Python
<H3> Algorithm:</H3>
Step 1:Import necessary libraries: pandas, networkx, matplotlib.pyplot, Bbn, Edge, EdgeType, BbnNode, Variable, EvidenceBuilder, InferenceController<br/>
Step 2:Set pandas options to display more columns<br/>
Step 3:Read in weather data from a CSV file using pandas<br/>
Step 4:Remove records where the target variable RainTomorrow has missing values<br/>
Step 5:Fill in missing values in other columns with the column mean<br/>
Step 6:Create bands for variables that will be used in the model (Humidity9amCat, Humidity3pmCat, and WindGustSpeedCat)<br/>
Step 7:Define a function to calculate probability distributions, which go into the Bayesian Belief Network (BBN)<br/>
Step 8:Create BbnNode objects for Humidity9amCat, Humidity3pmCat, WindGustSpeedCat, and RainTomorrow, using the probs() function to calculate their probabilities<br/>
Step 9:Create a Bbn object and add the BbnNode objects to it, along with edges between the nodes<br/>
Step 10:Convert the BBN to a join tree using the InferenceController<br/>
Step 11:Set node positions for the graph<br/>
Step 12:Set options for the graph appearance<br/>
Step 13:Generate the graph using networkx<br/>
Step 14:Update margins and display the graph using matplotlib.pyplot<br/>

## Program:
```py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from pybbn.graph.dag import Bbn
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.graph.edge import Edge, EdgeType
from pybbn.pptc.inferencecontroller import InferenceController

df = pd.read_csv("weatherAUS.csv")

df = df.dropna(subset=["RainTomorrow"])

df = df.fillna(df.mode().iloc[0])

df["H9"] = df["Humidity9am"].apply(
    lambda x: ">60" if x > 60 else "<=60"
)

df["H3"] = df["Humidity3pm"].apply(
    lambda x: ">60" if x > 60 else "<=60"
)

df["W"] = df["WindGustSpeed"].apply(
    lambda x: "<=40" if x <= 40 else "40-50" if x <= 50 else ">50"
)

p1 = pd.crosstab(
    df["H9"],
    columns="count",
    normalize=True
).values.flatten().tolist()

p2 = pd.crosstab(
    df["H9"],
    df["H3"],
    normalize="index"
).values.flatten().tolist()

p3 = pd.crosstab(
    df["W"],
    columns="count",
    normalize=True
).values.flatten().tolist()

p4 = pd.crosstab(
    [df["H3"], df["W"]],
    df["RainTomorrow"],
    normalize="index"
).values.flatten().tolist()

H9 = BbnNode(
    Variable(0, "H9", ["<=60", ">60"]),
    p1
)

H3 = BbnNode(
    Variable(1, "H3", ["<=60", ">60"]),
    p2
)

W = BbnNode(
    Variable(2, "W", ["<=40", "40-50", ">50"]),
    p3
)

RT = BbnNode(
    Variable(3, "RT", ["No", "Yes"]),
    p4
)

bbn = Bbn() \
    .add_node(H9) \
    .add_node(H3) \
    .add_node(W) \
    .add_node(RT) \
    .add_edge(Edge(H9, H3, EdgeType.DIRECTED)) \
    .add_edge(Edge(H3, RT, EdgeType.DIRECTED)) \
    .add_edge(Edge(W, RT, EdgeType.DIRECTED))

join_tree = InferenceController.apply(bbn)

print("RainTomorrow Probabilities:\n")

potential = join_tree.get_bbn_potential(RT)

for entry in potential.entries:
    print(entry)

g, labels = bbn.to_nx_graph()

pos = {
    0: (-1, 2),
    1: (-1, 0.5),
    2: (1, 0.5),
    3: (0, -1)
}

nx.draw_networkx(
    g,
    labels=labels,
    pos=pos,
    node_size=4000,
    node_color="pink",
    edge_color="green",
    font_size=14,
    width=3
)

plt.axis("off")
plt.show()

```

## Output:
<img width="898" height="702" alt="image" src="https://github.com/user-attachments/assets/f3388edd-a1ec-479d-bd70-5a47ff63e9ae" />

<img width="710" height="533" alt="image" src="https://github.com/user-attachments/assets/c93121e2-a6f7-42b8-8cdb-61e40a586482" />


## Result:
   Thus a Bayesian Network is generated using Python was executed Successfully.

