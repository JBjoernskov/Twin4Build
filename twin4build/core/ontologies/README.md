# Vendored ontologies

Local snapshots of the ontologies twin4build parses at runtime for
class-hierarchy reasoning during translation. They are the primary source;
the remote URLs (see `twin4build.core.ontology_remote`) are only used as a
fallback if a local file is missing. This keeps translation deterministic
and offline-capable - a failed ontology download used to silently degrade
signature-pattern matching and drop components from translated models
(https://github.com/JBjoernskov/Twin4Build/issues/114).

All files are serialized as turtle from the sources below (fetched
2026-08-04). To refresh, run `python twin4build/core/ontologies/refresh.py`
against the same pinned versions, or bump the versions deliberately in
`twin4build.core.ontology_remote` first.

| file | source | version | license |
|---|---|---|---|
| `saref.ttl` | https://saref.etsi.org/core/v3.1.1/ | v3.1.1 | ETSI OSL (BSD-3-style) |
| `saref4bldg.ttl` | https://saref.etsi.org/saref4bldg/v1.1.2/ | v1.1.2 | ETSI OSL (BSD-3-style) |
| `saref4syst.ttl` | https://saref.etsi.org/saref4syst/ | v1.1.2 | ETSI OSL (BSD-3-style) |
| `brick.ttl` | https://brickschema.org/schema/1.4.1/Brick.ttl | 1.4.1 | BSD-3-Clause |
| `fso.ttl` | https://alikucukavci.github.io/FSO/fso.ttl | latest | CC-BY 4.0 |
| `bot.ttl` | http://www.w3id.org/bot/bot.ttl | latest | CC-BY 4.0 |
| `rdf.ttl` | http://www.w3.org/1999/02/22-rdf-syntax-ns# | - | W3C |
| `rdfs.ttl` | http://www.w3.org/2000/01/rdf-schema# | - | W3C |
| `owl.ttl` | http://www.w3.org/2002/07/owl# | - | W3C |
| `rec.ttl` | https://w3id.org/rec# | latest | MIT |
