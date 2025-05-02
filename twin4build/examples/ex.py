# %pip install git+https://github.com/JBjoernskov/Twin4Build.git # Uncomment in google colab
import sys
sys.path.append(r"C:\Users\jabj\Documents\python\Twin4Build")

import time
start_time = time.time()

print("Importing twin4build")
import twin4build as tb
print("Took", time.time() - start_time, "seconds")
# import datetime
# from dateutil import tz
# import twin4build.utils.plot.plot as plot


turtle_file = r"C:\Users\jabj\OneDrive - Syddansk Universitet\excel\one_room_example_model.xlsm"
# turtle_file = r"C:\Users\jabj\OneDrive - Syddansk Universitet\excel\configuration_template_DP37_full_no_cooling.xlsm"
# turtle_file = r"https://brickschema.org/ttl/mortar/bldg1.ttl"

print("Creating semantic model")
start_time = time.time()
sem_model = tb.SemanticModel(turtle_file)#, additional_namespaces=namespaces)
print("Took", time.time() - start_time, "seconds")

print("Reasoning semantic model")
start_time = time.time()
# namespaces = ["https://alikucukavci.github.io/FSO/fso.ttl", tb.SAREF, tb.S4BLDG, tb.S4SYST]
namespaces = {"FSO": "https://alikucukavci.github.io/FSO/fso.ttl",
              "SAREF": tb.SAREF,
              "S4BLDG": tb.S4BLDG,
              "S4SYST": tb.S4SYST}
sem_model.reason(namespaces) # Adds any missing triples
print("Took", time.time() - start_time, "seconds")

# Define a query to filter the graph before visualizing it.
# Here, we remove all triples that includes predicates: rdf:type,
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

# query = """
# CONSTRUCT {
#     ?s ?p ?o 
# }
# WHERE {
#     ?s ?p ?o .
#     FILTER (?p = brick:feeds ||
#             ?p = brick:hasPart)
# }
# """

print("Visualizing semantic model")
start_time = time.time()
sem_model.visualize(query=query) # Visualize the semantic model
print("Took", time.time() - start_time, "seconds")




print("Translating semantic model")
start_time = time.time()
translator = tb.Translator()
sim_model = translator.translate(sem_model)
sim_model.visualize()
sim_model.load()
print("Took", time.time() - start_time, "seconds")

# model = tb.Model(id="model")
# model.load(semantic_model=sem_model)