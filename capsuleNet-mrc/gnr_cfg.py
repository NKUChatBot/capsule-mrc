from pycallgraph import PyCallGraph
from run import run


from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

with PyCallGraph(output=GraphvizOutput()):
	run()