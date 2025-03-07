# %pip install git+https://github.com/JBjoernskov/Twin4Build.git # Uncomment in google colab
import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")
import twin4build as tb
# import datetime
# from dateutil import tz
# import twin4build.utils.plot.plot as plot

    
turtle_file = r"C:\Users\jabj\OneDrive - Syddansk Universitet\excel\one_room_example_model.xlsm"
namespaces = ["https://alikucukavci.github.io/FSO/fso.ttl"]
sem_model = tb.SemanticModel(turtle_file, additional_namespaces=namespaces)
sem_model.reason(namespaces) # Adds any missing triples

# Define a query to filter the graph before visualizing it.
# Here, we remove all triples with predicates: rdf:type, 
# s4syst:subSystemOf, s4syst:hasSubSystem.
query = """
CONSTRUCT {
    ?s ?p ?o
}
WHERE {
    ?s ?p ?o .
    FILTER (?p != rdf:type && 
    ?p != s4syst:subSystemOf && 
    ?p != s4syst:hasSubSystem)
}
"""

sem_model.visualize(query=query) # Visualize the semantic model
translator = tb.Translator()
translator.translate(sem_model)