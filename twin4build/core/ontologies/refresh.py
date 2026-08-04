"""Refresh the vendored ontology snapshots in this directory.

Run from the repository root:

    python twin4build/core/ontologies/refresh.py

Each source is parsed with rdflib (same content negotiation as runtime
parsing used to do) and re-serialized as turtle, so the vendored triples are
exactly what a runtime download would have produced.  Sources are pinned in
:class:`twin4build.core.ontology_remote`; bump versions there deliberately.
"""

import os

import rdflib

from twin4build.core import ontology_remote

SOURCES = {
    "saref.ttl": ontology_remote.SAREF,
    "saref4bldg.ttl": ontology_remote.S4BLDG,
    "saref4syst.ttl": ontology_remote.S4SYST,
    "brick.ttl": ontology_remote.BRICK,
    "fso.ttl": ontology_remote.FSO,
    "bot.ttl": ontology_remote.BOT,
    "rdf.ttl": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs.ttl": "http://www.w3.org/2000/01/rdf-schema#",
    "owl.ttl": "http://www.w3.org/2002/07/owl#",
    "rec.ttl": "https://w3id.org/rec#",
}


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for fname, url in SOURCES.items():
        dest = os.path.join(out_dir, fname)
        try:
            g = rdflib.Graph()
            g.parse(url)
            g.serialize(destination=dest, format="turtle")
            size = os.path.getsize(dest)
            print(
                f"OK   {fname:16s} {len(g):6d} triples {size / 1024:8.1f} KiB"
                f"  <- {url}"
            )
        except Exception as e:
            print(f"FAIL {fname:16s} {url}: {e}")


if __name__ == "__main__":
    main()
